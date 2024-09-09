# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/userProfileRecommend/userProfileRecommend.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n5proto/userProfileRecommend/userProfileRecommend.proto\x12\x14userProfileRecommend\"0\n\x0eProfileRequest\x12\x10\n\x08memberId\x18\x01 \x01(\x03\x12\x0c\n\x04page\x18\x02 \x01(\x05\"\x97\x01\n\x0bSimilarItem\x12\x14\n\x0csong_info_id\x18\x01 \x01(\t\x12\x11\n\tsong_name\x18\x02 \x01(\t\x12\x13\n\x0b\x61rtist_name\x18\x03 \x01(\t\x12\n\n\x02mr\x18\x04 \x01(\x08\x12\x0c\n\x04ssss\x18\x05 \x01(\t\x12\x16\n\x0e\x61udio_file_url\x18\x06 \x01(\t\x12\x18\n\x10similarity_score\x18\x07 \x01(\x02\"K\n\x0fProfileResponse\x12\x38\n\rsimilar_items\x18\x01 \x03(\x0b\x32!.userProfileRecommend.SimilarItem2q\n\x0bUserProfile\x12\x62\n\x11\x43reateUserProfile\x12$.userProfileRecommend.ProfileRequest\x1a%.userProfileRecommend.ProfileResponse\"\x00\x62\x06proto3')



_PROFILEREQUEST = DESCRIPTOR.message_types_by_name['ProfileRequest']
_SIMILARITEM = DESCRIPTOR.message_types_by_name['SimilarItem']
_PROFILERESPONSE = DESCRIPTOR.message_types_by_name['ProfileResponse']
ProfileRequest = _reflection.GeneratedProtocolMessageType('ProfileRequest', (_message.Message,), {
  'DESCRIPTOR' : _PROFILEREQUEST,
  '__module__' : 'proto.userProfileRecommend.userProfileRecommend_pb2'
  # @@protoc_insertion_point(class_scope:userProfileRecommend.ProfileRequest)
  })
_sym_db.RegisterMessage(ProfileRequest)

SimilarItem = _reflection.GeneratedProtocolMessageType('SimilarItem', (_message.Message,), {
  'DESCRIPTOR' : _SIMILARITEM,
  '__module__' : 'proto.userProfileRecommend.userProfileRecommend_pb2'
  # @@protoc_insertion_point(class_scope:userProfileRecommend.SimilarItem)
  })
_sym_db.RegisterMessage(SimilarItem)

ProfileResponse = _reflection.GeneratedProtocolMessageType('ProfileResponse', (_message.Message,), {
  'DESCRIPTOR' : _PROFILERESPONSE,
  '__module__' : 'proto.userProfileRecommend.userProfileRecommend_pb2'
  # @@protoc_insertion_point(class_scope:userProfileRecommend.ProfileResponse)
  })
_sym_db.RegisterMessage(ProfileResponse)

_USERPROFILE = DESCRIPTOR.services_by_name['UserProfile']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _PROFILEREQUEST._serialized_start=79
  _PROFILEREQUEST._serialized_end=127
  _SIMILARITEM._serialized_start=130
  _SIMILARITEM._serialized_end=281
  _PROFILERESPONSE._serialized_start=283
  _PROFILERESPONSE._serialized_end=358
  _USERPROFILE._serialized_start=360
  _USERPROFILE._serialized_end=473
# @@protoc_insertion_point(module_scope)
