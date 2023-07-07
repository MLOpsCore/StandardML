from standardml.cloud.providers import *
from standardml.base import AbstractFactory


class CloudMounterFactory(AbstractFactory):
    cloud_types = {
        'aws': AWSMounter,
        'gcp': GCPMounter,
        'azure': AzureMounter,
        'ibm': IBMMounter
    }

    def _init_(self, cloud_type, bucket_name, mount_point):
        self.cloud_type = cloud_type

        self.bucket_name = bucket_name
        self.mount_point = mount_point

    def create(self) -> CloudMounter:
        return self.cloud_types[self.cloud_type](
            bucket_name=self.bucket_name,
            mount_point=self.mount_point
        )
