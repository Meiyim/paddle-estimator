# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: atarashi/data/feature.proto

import sys
_b = sys.version_info[0] < 3 and (lambda x: x) or (
    lambda x: x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor.FileDescriptor(
    name='atarashi/data/feature.proto',
    package='atarashi',
    syntax='proto3',
    serialized_options=None,
    serialized_pb=_b(
        '\n\x1b\x61tarashi/data/feature.proto\x12\x08\x61tarashi\"\x1a\n\tBytesList\x12\r\n\x05value\x18\x01 \x03(\x0c\"\x1e\n\tFloatList\x12\x11\n\x05value\x18\x01 \x03(\x02\x42\x02\x10\x01\"\x1e\n\tInt64List\x12\x11\n\x05value\x18\x01 \x03(\x03\x42\x02\x10\x01\"\x92\x01\n\x07\x46\x65\x61ture\x12)\n\nbytes_list\x18\x01 \x01(\x0b\x32\x13.atarashi.BytesListH\x00\x12)\n\nfloat_list\x18\x02 \x01(\x0b\x32\x13.atarashi.FloatListH\x00\x12)\n\nint64_list\x18\x03 \x01(\x0b\x32\x13.atarashi.Int64ListH\x00\x42\x06\n\x04kind\"\x7f\n\x08\x46\x65\x61tures\x12\x30\n\x07\x66\x65\x61ture\x18\x01 \x03(\x0b\x32\x1f.atarashi.Features.FeatureEntry\x1a\x41\n\x0c\x46\x65\x61tureEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12 \n\x05value\x18\x02 \x01(\x0b\x32\x11.atarashi.Feature:\x02\x38\x01\"1\n\x0b\x46\x65\x61tureList\x12\"\n\x07\x66\x65\x61ture\x18\x01 \x03(\x0b\x32\x11.atarashi.Feature\"\x98\x01\n\x0c\x46\x65\x61tureLists\x12=\n\x0c\x66\x65\x61ture_list\x18\x01 \x03(\x0b\x32\'.atarashi.FeatureLists.FeatureListEntry\x1aI\n\x10\x46\x65\x61tureListEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12$\n\x05value\x18\x02 \x01(\x0b\x32\x15.atarashi.FeatureList:\x02\x38\x01\x62\x06proto3'
    ))

_BYTESLIST = _descriptor.Descriptor(
    name='BytesList',
    full_name='atarashi.BytesList',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='value',
            full_name='atarashi.BytesList.value',
            index=0,
            number=1,
            type=12,
            cpp_type=9,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=41,
    serialized_end=67, )

_FLOATLIST = _descriptor.Descriptor(
    name='FloatList',
    full_name='atarashi.FloatList',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='value',
            full_name='atarashi.FloatList.value',
            index=0,
            number=1,
            type=2,
            cpp_type=6,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=_b('\020\001'),
            file=DESCRIPTOR),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=69,
    serialized_end=99, )

_INT64LIST = _descriptor.Descriptor(
    name='Int64List',
    full_name='atarashi.Int64List',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='value',
            full_name='atarashi.Int64List.value',
            index=0,
            number=1,
            type=3,
            cpp_type=2,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=_b('\020\001'),
            file=DESCRIPTOR),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=101,
    serialized_end=131, )

_FEATURE = _descriptor.Descriptor(
    name='Feature',
    full_name='atarashi.Feature',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='bytes_list',
            full_name='atarashi.Feature.bytes_list',
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='float_list',
            full_name='atarashi.Feature.float_list',
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='int64_list',
            full_name='atarashi.Feature.int64_list',
            index=2,
            number=3,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
        _descriptor.OneofDescriptor(
            name='kind',
            full_name='atarashi.Feature.kind',
            index=0,
            containing_type=None,
            fields=[]),
    ],
    serialized_start=134,
    serialized_end=280, )

_FEATURES_FEATUREENTRY = _descriptor.Descriptor(
    name='FeatureEntry',
    full_name='atarashi.Features.FeatureEntry',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='key',
            full_name='atarashi.Features.FeatureEntry.key',
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=_b("").decode('utf-8'),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='value',
            full_name='atarashi.Features.FeatureEntry.value',
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=_b('8\001'),
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=344,
    serialized_end=409, )

_FEATURES = _descriptor.Descriptor(
    name='Features',
    full_name='atarashi.Features',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='feature',
            full_name='atarashi.Features.feature',
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR),
    ],
    extensions=[],
    nested_types=[_FEATURES_FEATUREENTRY, ],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=282,
    serialized_end=409, )

_FEATURELIST = _descriptor.Descriptor(
    name='FeatureList',
    full_name='atarashi.FeatureList',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='feature',
            full_name='atarashi.FeatureList.feature',
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=411,
    serialized_end=460, )

_FEATURELISTS_FEATURELISTENTRY = _descriptor.Descriptor(
    name='FeatureListEntry',
    full_name='atarashi.FeatureLists.FeatureListEntry',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='key',
            full_name='atarashi.FeatureLists.FeatureListEntry.key',
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=_b("").decode('utf-8'),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='value',
            full_name='atarashi.FeatureLists.FeatureListEntry.value',
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=_b('8\001'),
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=542,
    serialized_end=615, )

_FEATURELISTS = _descriptor.Descriptor(
    name='FeatureLists',
    full_name='atarashi.FeatureLists',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='feature_list',
            full_name='atarashi.FeatureLists.feature_list',
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR),
    ],
    extensions=[],
    nested_types=[_FEATURELISTS_FEATURELISTENTRY, ],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=463,
    serialized_end=615, )

_FEATURE.fields_by_name['bytes_list'].message_type = _BYTESLIST
_FEATURE.fields_by_name['float_list'].message_type = _FLOATLIST
_FEATURE.fields_by_name['int64_list'].message_type = _INT64LIST
_FEATURE.oneofs_by_name['kind'].fields.append(_FEATURE.fields_by_name[
    'bytes_list'])
_FEATURE.fields_by_name[
    'bytes_list'].containing_oneof = _FEATURE.oneofs_by_name['kind']
_FEATURE.oneofs_by_name['kind'].fields.append(_FEATURE.fields_by_name[
    'float_list'])
_FEATURE.fields_by_name[
    'float_list'].containing_oneof = _FEATURE.oneofs_by_name['kind']
_FEATURE.oneofs_by_name['kind'].fields.append(_FEATURE.fields_by_name[
    'int64_list'])
_FEATURE.fields_by_name[
    'int64_list'].containing_oneof = _FEATURE.oneofs_by_name['kind']
_FEATURES_FEATUREENTRY.fields_by_name['value'].message_type = _FEATURE
_FEATURES_FEATUREENTRY.containing_type = _FEATURES
_FEATURES.fields_by_name['feature'].message_type = _FEATURES_FEATUREENTRY
_FEATURELIST.fields_by_name['feature'].message_type = _FEATURE
_FEATURELISTS_FEATURELISTENTRY.fields_by_name[
    'value'].message_type = _FEATURELIST
_FEATURELISTS_FEATURELISTENTRY.containing_type = _FEATURELISTS
_FEATURELISTS.fields_by_name[
    'feature_list'].message_type = _FEATURELISTS_FEATURELISTENTRY
DESCRIPTOR.message_types_by_name['BytesList'] = _BYTESLIST
DESCRIPTOR.message_types_by_name['FloatList'] = _FLOATLIST
DESCRIPTOR.message_types_by_name['Int64List'] = _INT64LIST
DESCRIPTOR.message_types_by_name['Feature'] = _FEATURE
DESCRIPTOR.message_types_by_name['Features'] = _FEATURES
DESCRIPTOR.message_types_by_name['FeatureList'] = _FEATURELIST
DESCRIPTOR.message_types_by_name['FeatureLists'] = _FEATURELISTS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

BytesList = _reflection.GeneratedProtocolMessageType(
    'BytesList',
    (_message.Message, ),
    dict(
        DESCRIPTOR=_BYTESLIST,
        __module__='atarashi.data.feature_pb2'
        # @@protoc_insertion_point(class_scope:atarashi.BytesList)
    ))
_sym_db.RegisterMessage(BytesList)

FloatList = _reflection.GeneratedProtocolMessageType(
    'FloatList',
    (_message.Message, ),
    dict(
        DESCRIPTOR=_FLOATLIST,
        __module__='atarashi.data.feature_pb2'
        # @@protoc_insertion_point(class_scope:atarashi.FloatList)
    ))
_sym_db.RegisterMessage(FloatList)

Int64List = _reflection.GeneratedProtocolMessageType(
    'Int64List',
    (_message.Message, ),
    dict(
        DESCRIPTOR=_INT64LIST,
        __module__='atarashi.data.feature_pb2'
        # @@protoc_insertion_point(class_scope:atarashi.Int64List)
    ))
_sym_db.RegisterMessage(Int64List)

Feature = _reflection.GeneratedProtocolMessageType(
    'Feature',
    (_message.Message, ),
    dict(
        DESCRIPTOR=_FEATURE,
        __module__='atarashi.data.feature_pb2'
        # @@protoc_insertion_point(class_scope:atarashi.Feature)
    ))
_sym_db.RegisterMessage(Feature)

Features = _reflection.GeneratedProtocolMessageType(
    'Features',
    (_message.Message, ),
    dict(
        FeatureEntry=_reflection.GeneratedProtocolMessageType(
            'FeatureEntry',
            (_message.Message, ),
            dict(
                DESCRIPTOR=_FEATURES_FEATUREENTRY,
                __module__='atarashi.data.feature_pb2'
                # @@protoc_insertion_point(class_scope:atarashi.Features.FeatureEntry)
            )),
        DESCRIPTOR=_FEATURES,
        __module__='atarashi.data.feature_pb2'
        # @@protoc_insertion_point(class_scope:atarashi.Features)
    ))
_sym_db.RegisterMessage(Features)
_sym_db.RegisterMessage(Features.FeatureEntry)

FeatureList = _reflection.GeneratedProtocolMessageType(
    'FeatureList',
    (_message.Message, ),
    dict(
        DESCRIPTOR=_FEATURELIST,
        __module__='atarashi.data.feature_pb2'
        # @@protoc_insertion_point(class_scope:atarashi.FeatureList)
    ))
_sym_db.RegisterMessage(FeatureList)

FeatureLists = _reflection.GeneratedProtocolMessageType(
    'FeatureLists',
    (_message.Message, ),
    dict(
        FeatureListEntry=_reflection.GeneratedProtocolMessageType(
            'FeatureListEntry',
            (_message.Message, ),
            dict(
                DESCRIPTOR=_FEATURELISTS_FEATURELISTENTRY,
                __module__='atarashi.data.feature_pb2'
                # @@protoc_insertion_point(class_scope:atarashi.FeatureLists.FeatureListEntry)
            )),
        DESCRIPTOR=_FEATURELISTS,
        __module__='atarashi.data.feature_pb2'
        # @@protoc_insertion_point(class_scope:atarashi.FeatureLists)
    ))
_sym_db.RegisterMessage(FeatureLists)
_sym_db.RegisterMessage(FeatureLists.FeatureListEntry)

_FLOATLIST.fields_by_name['value']._options = None
_INT64LIST.fields_by_name['value']._options = None
_FEATURES_FEATUREENTRY._options = None
_FEATURELISTS_FEATURELISTENTRY._options = None
# @@protoc_insertion_point(module_scope)