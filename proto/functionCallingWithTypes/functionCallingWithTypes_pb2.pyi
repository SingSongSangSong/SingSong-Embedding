from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class FunctionCallingWithTypesRequest(_message.Message):
    __slots__ = ("memberId", "gender", "year", "command")
    MEMBERID_FIELD_NUMBER: _ClassVar[int]
    GENDER_FIELD_NUMBER: _ClassVar[int]
    YEAR_FIELD_NUMBER: _ClassVar[int]
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    memberId: int
    gender: str
    year: str
    command: str
    def __init__(self, memberId: _Optional[int] = ..., gender: _Optional[str] = ..., year: _Optional[str] = ..., command: _Optional[str] = ...) -> None: ...

class FunctionCallingWithTypesResponse(_message.Message):
    __slots__ = ("songInfoId",)
    SONGINFOID_FIELD_NUMBER: _ClassVar[int]
    songInfoId: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, songInfoId: _Optional[_Iterable[int]] = ...) -> None: ...
