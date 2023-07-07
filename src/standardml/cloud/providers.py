import subprocess
from typing import Optional

from standardml.cloud.base import CloudMounter, CloudMounterConfig


class AWSMounterConfig(CloudMounterConfig):
    region: Optional[str]


class AWSMounter(CloudMounter):
    """
    Mounter for AWS S3.
    """
    def mount(self, config: AWSMounterConfig):
        cmd = f's3fs {self.bucket_name} {self.mount_point} ' \
              f'-o iam_role=auto -o allow_other'
        if config.region:
            cmd += f' -o endpoint={config.region}.amazonaws.com'
        subprocess.run(cmd.split())


class GCPMounterConfig(CloudMounterConfig):
    ...


class GCPMounter(CloudMounter):
    """
    Mounter for GCP GCS.
    """
    def mount(self, config: GCPMounterConfig):
        cmd = f'gcsfuse {self.bucket_name} {self.mount_point}'
        subprocess.run(cmd.split())


class AzureMounterConfig(CloudMounterConfig):
    account_name: str
    account_key: str


class AzureMounter(CloudMounter):
    """
    Mounter for Azure Blob.
    """
    def mount(self, config: AzureMounterConfig):
        cmd = f'blobfuse {self.mount_point} ' \
              f'--container-name={self.bucket_name} ' \
              f'--account-name={config.account_name} ' \
              f'--account-key={config.account_key}'
        subprocess.run(cmd.split())


class IBMMounterConfig(CloudMounterConfig):
    region: str
    creds_file: str


class IBMMounter(CloudMounter):
    """
    Mounter for IBM COS.
    """
    def mount(self, config: IBMMounterConfig):
        cmd = f's3fs {self.bucket_name} {self.mount_point} ' \
              f'-o url=https://{config.region}.cloud-object-storage.appdomain.cloud ' \
              f'-o passwd_file={config.creds_file} -o allow_other'
        subprocess.run(cmd.split())
