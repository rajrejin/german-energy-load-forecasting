"""Azure Blob Storage store (also used against the Azurite emulator)."""

from __future__ import annotations

from typing import Any

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient, ContainerClient

from energy.store.base import ArtifactStore, KeyNotFoundError

# Public, documented development account of the Azurite emulator (not a secret).
AZURITE_CONNECTION_STRING = (
    "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;"
    "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;"
    "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
)


class AzureBlobStore(ArtifactStore):
    def __init__(
        self,
        container: str = "energy",
        connection_string: str | None = None,
        account_url: str | None = None,
        credential: Any = None,
    ):
        if connection_string:
            svc = BlobServiceClient.from_connection_string(connection_string)
        elif account_url:
            if credential is None:
                from azure.identity import DefaultAzureCredential

                credential = DefaultAzureCredential()
            svc = BlobServiceClient(account_url=account_url, credential=credential)
        else:
            raise ValueError("connection_string or account_url is required")
        self.container: ContainerClient = svc.get_container_client(container)
        try:
            self.container.create_container()
        except ResourceExistsError:
            pass

    @classmethod
    def azurite(
        cls, container: str = "energy", connection_string: str | None = None
    ) -> AzureBlobStore:
        return cls(
            container=container, connection_string=connection_string or AZURITE_CONNECTION_STRING
        )

    def put_bytes(self, key: str, data: bytes) -> None:
        self.container.upload_blob(key, data, overwrite=True)

    def get_bytes(self, key: str) -> bytes:
        try:
            return self.container.download_blob(key).readall()  # type: ignore[no-any-return]
        except ResourceNotFoundError as e:
            raise KeyNotFoundError(key) from e

    def exists(self, key: str) -> bool:
        return bool(self.container.get_blob_client(key).exists())

    def list_keys(self, prefix: str = "") -> list[str]:
        return sorted(b.name for b in self.container.list_blobs(name_starts_with=prefix))

    def delete(self, key: str) -> None:
        try:
            self.container.delete_blob(key)
        except ResourceNotFoundError:
            pass
