from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

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

class SongInfo(_message.Message):
    __slots__ = ("songNumber", "songName", "artistName", "songInfoId", "album", "isMr", "isLive", "melonSongId")
    SONGNUMBER_FIELD_NUMBER: _ClassVar[int]
    SONGNAME_FIELD_NUMBER: _ClassVar[int]
    ARTISTNAME_FIELD_NUMBER: _ClassVar[int]
    SONGINFOID_FIELD_NUMBER: _ClassVar[int]
    ALBUM_FIELD_NUMBER: _ClassVar[int]
    ISMR_FIELD_NUMBER: _ClassVar[int]
    ISLIVE_FIELD_NUMBER: _ClassVar[int]
    MELONSONGID_FIELD_NUMBER: _ClassVar[int]
    songNumber: int
    songName: str
    artistName: str
    songInfoId: int
    album: str
    isMr: bool
    isLive: bool
    melonSongId: str
    def __init__(self, songNumber: _Optional[int] = ..., songName: _Optional[str] = ..., artistName: _Optional[str] = ..., songInfoId: _Optional[int] = ..., album: _Optional[str] = ..., isMr: bool = ..., isLive: bool = ..., melonSongId: _Optional[str] = ...) -> None: ...

class FunctionCallingWithTypesResponse(_message.Message):
    __slots__ = ("songInfos", "message")
    SONGINFOS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    songInfos: _containers.RepeatedCompositeFieldContainer[SongInfo]
    message: str
    def __init__(self, songInfos: _Optional[_Iterable[_Union[SongInfo, _Mapping]]] = ..., message: _Optional[str] = ...) -> None: ...
