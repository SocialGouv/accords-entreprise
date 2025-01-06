from typing import Literal, Union

AccordEntrepriseID = str
ChunkID = int
ChunkMetadata = dict
ChunkStatus = Literal["UP_TO_DATE", "OUTDATED", "DELETED"]
ChunkText = str
Distance = float
DocumentID = Union[AccordEntrepriseID]
DocumentName = str
DocumentText = str
Embedding = list[float]
MetadataVersion = int
TimestampSecond = int
