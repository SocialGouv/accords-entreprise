from typing import Literal, Union

AccordEntrepriseID = str
ChunkID = int
ChunkMetadata = dict
ChunkText = str
DocumentText = str
DocumentID = Union[AccordEntrepriseID]
DocumentName = str
Embedding = list[float]
TimestampSecond = int
MetadataVersion = int
ChunkStatus = Literal["UP_TO_DATE", "OUTDATED", "DELETED"]
