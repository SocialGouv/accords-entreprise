from typing import Literal, Union

AccordEntrepriseID = str
ChunkID = int
ChunkMetadata = dict
ChunkStatus = Literal["UP_TO_DATE", "OUTDATED", "DELETED"]
Distance = float
DocumentID = Union[AccordEntrepriseID]
DocumentName = str
Embeddings = list[float]
MetadataVersion = int
TimestampSecond = int
