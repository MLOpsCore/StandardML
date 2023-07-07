import abc

from pydantic import BaseModel


class CloudMounterConfig(BaseModel):
    ...


class CloudMounter(BaseModel, metaclass=abc.ABCMeta):
    """
    Base class for cloud mounters.
    """

    bucket_name: str
    mount_point: str

    @abc.abstractmethod
    def mount(self, config: CloudMounterConfig):
        pass
