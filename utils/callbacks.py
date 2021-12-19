import os
import subprocess
import shutil

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only


class VersionedCallback(Callback):
    def __init__(self, save_root):
        self.save_root = save_root

    @property
    def version(self) -> int:
        """Get the experiment version.

        Returns:
            The experiment version if specified else the next version.
        """
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self):
        existing_versions = []
        if os.path.isdir(self.save_root):
            for f in os.listdir(self.save_root):
                bn = os.path.basename(f)
                if bn.startswith("version_"):
                    dir_ver = os.path.splitext(bn)[0].split("_")[1].replace("/", "")
                    existing_versions.append(int(dir_ver))
        if len(existing_versions) == 0:
            return 0
        return max(existing_versions) + 1


class CodeSnapshotCallback(VersionedCallback):
    def __init__(self, save_root, version=None):
        super().__init__(save_root)
        self._version = version
    
    def get_file_list(self):
        return [
            b.decode() for b in
            set(subprocess.check_output('git ls-files', shell=True).splitlines()) |
            set(subprocess.check_output('git ls-files --others --exclude-standard', shell=True).splitlines())
        ]
    
    @rank_zero_only
    def save_code_snapshot(self):
        if os.path.exists(self.savedir):
            return
        os.makedirs(self.savedir)
        for f in self.get_file_list():
            if not os.path.exists(f):
                continue
            os.makedirs(os.path.join(self.savedir, os.path.dirname(f)), exist_ok=True)
            shutil.copyfile(f, os.path.join(self.savedir, f))

    def on_fit_start(self, trainer, pl_module):
        print('on fit start')
        self.save_code_snapshot()
    
    @property
    def savedir(self):
        return os.path.join(self.save_root, self.version if isinstance(self.version, str) else f"version_{self.version}")


class ConfigSnapshotCallback(VersionedCallback):
    def __init__(self, save_root, config, version=None):
        super().__init__(save_root)
        self.config = config
        self._version = version

    @rank_zero_only
    def save_config_snapshot(self):
        if os.path.exists(self.savepath):
            return
        os.makedirs(self.save_root, exist_ok=True)
        self.config.export(self.savepath)

    def on_pretrain_routine_start(self, trainer, pl_module):
        self.save_config_snapshot()
    
    @property
    def savepath(self):
        return os.path.join(self.save_root, self.version + '.yaml' if isinstance(self.version, str) else f"version_{self.version}.yaml")


class StorageCallback(Callback):
    def __init__(self):
        pass
