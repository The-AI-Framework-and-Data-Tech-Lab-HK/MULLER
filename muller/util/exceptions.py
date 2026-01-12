# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This file was originally part of Hub (now Deep Lake) project: https://github.com/activeloopai/deeplake/tree/release/2.8.5
# Commit: https://github.com/activeloopai/deeplake/tree/94c5e100292c164b80132baf741ef233dd41f3d7
# Source: https://github.com/activeloopai/deeplake/blob/94c5e100292c164b80132baf741ef233dd41f3d7/hub/util/exceptions.py
#
# Modifications Copyright (c) 2026 Bingyu Liu

from typing import Any, List, Sequence, Tuple, Optional, Union

import numpy as np

import muller


class InvalidBytesRequestedError(Exception):
    def __init__(self):
        super().__init__(
            "The byte range provided is invalid. Ensure that start_byte <= end_byte and start_byte > 0 and end_byte > 0"
        )


class DirectoryAtPathException(Exception):
    def __init__(self):
        super().__init__(
            "The provided path is incorrect for this operation, there is a directory at the path. "
            "Provide a path to a file."
        )


class LockedException(Exception):
    def __init__(self, message="The resource is currently locked."):
        super().__init__(message)


class FileAtPathException(Exception):
    def __init__(self, path):
        super().__init__(
            f"Expected a directory at path {path} but found a file instead."
        )


class PathNotEmptyException(Exception):
    def __init__(self, use_cloud=True):
        if use_cloud:
            super().__init__(
                f"Please use a url that points to an existing MULLER-F Dataset or an empty folder. If you wish to "
                f"delete the folder and its contents, you may run muller.delete(dataset_path, force=True)."
            )
        else:
            super().__init__(
                f"Specified path is not empty. If you wish to delete the folder and its contents, "
                f"you may run muller.delete(path, force=True)."
            )


class TransformError(Exception):
    def __init__(self, index=None, sample=None, samples_processed=0, suggest=False, is_batch=False):
        self.index = index
        self.sample = sample
        self.suggest = suggest
        # multiprocessing re raises error with str
        if isinstance(index, str):
            super().__init__(index)
        else:
            print_item = print_path = False
            if sample is not None:
                print_item = is_primitive(sample)
                print_path = has_path(sample)

            msg = f"Transform failed"
            if index is not None:
                msg += f" at index {index} of the input data"
                if is_batch:
                    msg += f"batch"

            if print_item:
                msg += f" on the item: {get_truncated_sample(sample)}"
            elif print_path:
                msg += f" on the sample at path: '{sample.path}'"
            msg += "."

            if samples_processed > 0:
                msg += f" Last checkpoint: {samples_processed} samples processed. " \
                       f"You can slice the input to resume from this point."

            msg += " See traceback for more details."

            if suggest:
                msg += (
                    " If you wish to skip the samples that cause errors,"
                    " please specify `ignore_errors=True`."
                )

            super().__init__(msg)


class InvalidTransformDataset(TransformError):
    def __init__(
            self,
            message="The TransformDataset (2nd argument to transform function) of one of the functions is invalid. "
                    "All the tensors should have equal length for it to be valid.",
    ):
        super().__init__(message)


class TensorMismatchError(TransformError):
    def __init__(self, tensors, output_keys, skip_ok=False):
        if skip_ok:
            super().__init__(
                f"One or more tensors generated during MULLER_F compute don't exist in the target dataset. "
                f"With skip_ok=True, you can skip certain tensors in the transform, "
                f"however you need to ensure that all tensors generated exist in the dataset.\n "
                f"Tensors in target dataset: {tensors}\n Tensors in output sample: {output_keys}"
            )
        else:
            super().__init__(
                f"One or more of the outputs generated during transform contain different tensors than "
                f"the ones present in the target dataset of transform.\n "
                f"Tensors in target dataset: {tensors}\n Tensors in output sample: {output_keys}. "
                f"If you want to do this, pass skip_ok=True to the eval method."
            )


class ReadOnlyModeError(Exception):
    def __init__(self, custom_message: Optional[str] = None):
        if custom_message is None:
            custom_message = "Modification when in read-only mode is not supported!"
        super().__init__(custom_message)


class ProviderSizeListMismatch(Exception):
    def __init__(self):
        super().__init__("Ensure that len(size_list) + 1 == len(provider_list)")


class ProviderListEmptyError(Exception):
    def __init__(self):
        super().__init__(
            "The provider_list passed to get_cache_chain needs to have 1 or more elements."
        )


class InvalidDatasetNameException(Exception):
    def __init__(self, path_type):
        if path_type == "local":
            message = "Local dataset names can only contain letters, numbers, spaces, `-`, `_` and `.`."
        else:
            message = "Please specify a dataset name that contains only letters, numbers, hyphens and underscores."
        super().__init__(message)


class UserNotLoggedInException(Exception):
    def __init__(self):
        message = (
            "You are not logged in to the target platform. Please double check"
        )
        super().__init__(message)


class MetaError(Exception):
    pass


class TensorMetaInvalidHtype(MetaError):
    def __init__(self, htype: str, available_htypes: Sequence[str]):
        super().__init__(
            f"Htype '{htype}' does not exist. Available htypes: {str(available_htypes)}"
        )


class TensorMetaInvalidHtypeOverwriteValue(MetaError):
    def __init__(self, key: str, value: Any, explanation: str = ""):
        super().__init__(
            f"Invalid value '{value}' for tensor meta key '{key}'. {explanation}"
        )


class TensorMetaInvalidHtypeOverwriteKey(MetaError):
    def __init__(self, htype: str, key: str, available_keys: Sequence[str]):
        super().__init__(
            f"Htype '{htype}' doesn't have a key for '{key}'. Available keys: {str(available_keys)}"
        )


class TensorMetaMissingRequiredValue(MetaError):
    def __init__(self, htype: str, key: Union[str, List[str]]):
        extra = ""
        if key == "sample_compression":
            extra = f"`sample_compression` may be `None` if you want your '{htype}' data to be uncompressed. " \
                    f"Available compressors: {muller.compressions}"

        if isinstance(key, list):
            message = f"Htype '{htype}' requires you to specify either one of {key} inside the `create_tensor` " \
                      f"method call. {extra}"
        else:
            message = f"Htype '{htype}' requires you to specify '{key}' inside the `create_tensor` method call. {extra}"
        super().__init__(message)


class TensorMetaMutuallyExclusiveKeysError(MetaError):
    def __init__(
            self, keys: Optional[List[str]] = None, custom_message: Optional[str] = None
    ):
        if custom_message:
            msg = custom_message
        else:
            msg = f"Following fields are mutually exclusive: {keys}. "
        super().__init__(msg)


class TensorInvalidSampleShapeError(Exception):
    def __init__(self, shape: Sequence[int], expected_dims: int):
        super().__init__(
            f"Sample shape length is expected to be {expected_dims}, actual length is {len(shape)}. "
            f"Full incoming shape: {shape}"
        )


class CompressionError(Exception):
    pass


class OutOfChunkCountError(Exception):
    pass


class OutOfSampleCountError(Exception):
    pass


class ChunkEngineError(Exception):
    pass


class ChunkIdEncoderError(ChunkEngineError):
    pass


class DatasetViewSavingError(Exception):
    pass


class SampleDecompressionError(CompressionError):
    def __init__(self, msg: Optional[str] = None, path: Optional[str] = None):
        message = "Could not decompress sample"
        if path:
            message += f" at {path}"
        message += f". Either the sample's buffer is corrupted, or it is in an unsupported format."
        message += f"Raw error output: '{msg}'."
        super().__init__(message)


class SampleCompressionError(CompressionError):
    def __init__(
            self,
            sample_shape: Tuple[int, ...],
            compression_format: Optional[str],
            message: str,
    ):
        super().__init__(
            f"Could not compress a sample with shape {str(sample_shape)} into '{compression_format}'. "
            f"Raw error output: '{message}'.",
        )


class SampleExtendingError(Exception):
    def __init__(self):
        message = (
            "Cannot extend because tensor(s) are not specified. Expected input to ds.extend is a dictionary. "
            "To extend tensors, you need to either specify the tensors and add the samples as a dictionary, like: "
            "`ds.extend({'image_tensor': samples, 'label_tensor': samples})` "
            "or you need to call `extend` method of the required tensor, "
            "like: `ds.image_tensor.extend(samples)`"
        )
        super().__init__(message)


class CorruptedSampleError(Exception):
    def __init__(self, compression, path: Optional[str] = None):
        message = f"Unable to decompress {compression} file"
        if path is not None:
            message += f" at {path}"
        message += "."
        super().__init__(message)


class SampleReadError(Exception):
    def __init__(self, path: str):
        super().__init__(f"Unable to read sample from {path}")


class UnableToReadFromUrlError(Exception):
    def __init__(self, url, status_code):
        super().__init__(f"Unable to read from url {url}. Status code: {status_code}")


class EmptyTensorError(Exception):
    def __init__(self, message):
        super().__init__(message)


class ReadSampleFromChunkError(Exception):
    def __init__(
            self,
            chunk_key: Optional[str],
            global_index: Optional[int] = None,
            tensor_name: Optional[str] = None,
    ):
        self.chunk_key = chunk_key
        message = "Unable to read sample"
        if global_index is not None:
            message += f" at index {global_index}"
        message += " from chunk"
        if chunk_key is not None:
            message += f" '{chunk_key}'"
        if tensor_name is not None:
            message += f" in tensor {tensor_name}"
        message += "."
        super().__init__(message)


class TensorDoesNotExistError(KeyError, AttributeError):
    def __init__(self, tensor_name: str):
        super().__init__(f"Tensor '{tensor_name}' does not exist.")


class UnsupportedInvertedIndexError(Exception):
    def __init__(self, htype, dtype):
        super().__init__(f"Unsupported inverted index htype {htype} and dtype {dtype}")


class UnsupportedArrowConvertError(Exception):
    def __init__(self, htype, dtype):
        super().__init__(f"Unsupported arrow convert htype {htype} and dtype {dtype}")


class UnsupportedMethod(Exception):
    def __init__(self, msg="Unsupported method yet."):
        super().__init__(msg)


class VersionControlError(Exception):
    pass


class RenameError(Exception):
    def __init__(self, msg="Only name of the dataset can be different in new path."):
        super().__init__(msg)


class RenameStorageError(Exception):
    def __init__(self, msg="Currently we only accept rename operation in LocalProvider."):
        super().__init__(msg)


class CheckoutError(VersionControlError):
    pass


class CorruptedMetaError(Exception):
    pass


class CouldNotCreateNewDatasetException(Exception):
    def __init__(
            self,
            path: str,
    ):
        message = f"Dataset at '{path}' doesn't exist, and you have no permissions to create one there. " \
                  f"Maybe a typo?"
        super().__init__(message)


class DatasetCorruptError(Exception):
    def __init__(self, message, action="", cause=None):
        self.message = message
        self.action = action
        self.__cause__ = cause

        super().__init__(self.message + (" " + self.action if self.action else ""))


class DatasetCorruptionError(Exception):
    def __init__(self):
        super().__init__(
            "Exception occurred (see Traceback). The dataset may be corrupted. "
            "Try using `reset=True` to reset HEAD changes and load the previous commit."
        )


class DatasetCreationError(Exception):
    def __init__(self, dataset_path):
        super().__init__(
            f"The dataset path {dataset_path} is invalid. "
            f"muller.dataset does not accept version address when writing a dataset."
        )


class DatasetHandlerError(Exception):
    def __init__(self, message):
        super().__init__(message)


class DatasetAlreadyExistsError(Exception):
    def __init__(self, path):
        super().__init__(f"A dataset already exists at the given path ({path}).")


class DatasetNotExistsError(Exception):
    def __init__(self, path):
        super().__init__(f"A MULLER-F dataset does not exist at the given path ({path}). "
                         f"Check the path provided or in case you want to create a new dataset, "
                         f"use muller.dataset() or muller.empty().")


class DatasetViewDeletionError(Exception):
    def __init__(self):
        super().__init__("Deleting managed views by path is not supported. Load the source dataset and "
                         "do `ds.delete_view(id)` instead.")


class DynamicTensorNumpyError(Exception):
    def __init__(self, key: str, index, property_key: str):
        super().__init__(
            f"Tensor '{key}' with index = {str(index)} has dynamic '{property_key}' and "
            f"cannot be converted into a `np.ndarray`. Try setting the parameter `aslist=True`"
        )


class GetChunkError(Exception):
    def __init__(
            self,
            chunk_key: Optional[str],
            global_index: Optional[int] = None,
            tensor_name: Optional[str] = None,
    ):
        self.chunk_key = chunk_key
        message = "Unable to get chunk"
        if chunk_key is not None:
            message += f" '{chunk_key}'"
        if global_index is not None:
            message += f" while retrieving data at index {global_index}"
        if tensor_name is not None:
            message += f" in tensor {tensor_name}"
        message += "."
        super().__init__(message)


class InvalidKeyTypeError(TypeError):
    def __init__(self, item: Any):
        super().__init__(
            f"Item '{str(item)}' of type '{type(item).__name__}' is not a valid key."
        )


class TensorAlreadyExistsError(Exception):
    def __init__(self, key: str):
        super().__init__(
            f"Tensor '{key}' already exists. You can use the `exist_ok=True` parameter to ignore this error message."
        )


class InvalidTensorNameError(Exception):
    def __init__(self, name: str):
        if name:
            msg = (
                f"The use of a reserved attribute '{name}' as a tensor name is invalid."
            )
        else:
            msg = f"Tensor name cannot be empty."
        super().__init__(msg)


class AgreementError(Exception):
    pass


class ExecuteError(Exception):
    pass


class MemoryDatasetCanNotBePickledError(Exception):
    def __init__(self):
        super().__init__(
            "Dataset having MemoryProvider as underlying storage should not be pickled as data won't be saved."
        )


class CommitError(VersionControlError):
    pass


class InfoError(Exception):
    pass


class EmptyCommitError(CommitError):
    pass


class CykhashPutError(Exception):
    pass


class CykhashGetError(Exception):
    pass


class CykhashLoadError(Exception):
    pass


class InvalidOperationError(Exception):
    def __init__(self, method: str, my_type: str):
        if method == "read_only":
            super().__init__("read_only property cannot be toggled for a dataset view.")
        else:
            super().__init__(f"{method} method cannot be called on a {my_type} view.")


class InvalidPermissionError(Exception):
    def __init__(self, response_code: int):
        if response_code == 200:
            super().__init__("Permissions are not all True.")
        else:
            super().__init__(f"Failed to fetch permissions. Status code: {response_code}")


def is_primitive(sample):
    """Determine whether the sample is primitive type"""
    if isinstance(sample, (str, int, float, bool)):
        return True
    if isinstance(sample, dict):
        for x, y in sample.items():
            if not is_primitive(x) or not is_primitive(y):
                return False
        return True
    if isinstance(sample, (list, tuple)):
        for x in sample:
            if not is_primitive(x):
                return False
        return True
    return False


def get_truncated_sample(sample, max_half_len=50):
    """Obtain truncated sample"""
    if len(str(sample)) > max_half_len * 2:
        return (
                str(sample)[:max_half_len] + "..." + str(sample)[int(-max_half_len - 1):]
        )
    return str(sample)


def has_path(sample):
    """Determine whether the sample is path"""
    from muller.core.sample import Sample

    return isinstance(sample, Sample) and sample.path is not None


class UnsupportedSchedulerError(TransformError):
    def __init__(self, scheduler):
        super().__init__(
            f"MULLER_F compute currently doesn't support {scheduler} scheduler."
        )


class SampleAppendError(Exception):
    def __init__(self, tensor, sample=None):
        print_item = print_path = False
        if sample is not None:
            print_item = is_primitive(sample)
            print_path = has_path(sample)
        if print_item or print_path:
            msg = "Failed to append the sample "

            if print_item:
                msg += str(sample) + " "
            elif print_path:
                msg += f"at path '{sample.path}' "
        else:
            msg = f"Failed to append a sample "

        msg += f"to the tensor '{tensor}'. See more details in the traceback."

        super().__init__(msg)


class InvalidInputDataError(TransformError):
    def __init__(self, operation):
        super().__init__(
            f"The data_in to transform is invalid. It doesn't support {operation} operation. "
            "Please use a list, a MULLER_F dataset or an object that supports both __getitem__ and __len__. "
            "Generators are not supported."
        )


class AllSamplesSkippedError(Exception):
    def __init__(self):
        super().__init__(
            "All samples were skipped during the transform. "
            "Ensure your transform pipeline is correct before you set `ignore_errors=True`."
        )


class InvalidOutputDatasetError(TransformError):
    def __init__(
            self, message="The output Dataset to transform should not be `read_only`."
    ):
        super().__init__(message)


class SampleAppendingError(Exception):
    def __init__(self):
        message = """Cannot append sample because tensor(s) are not specified.
        Expected input to ds.append is a dictionary. To append samples, you need to either specify the tensors 
        and append the samples as a dictionary, like: `ds.append({"image_tensor": sample, "label_tensor": sample})` 
        or you need to call `append` method of the required tensor, like: `ds.image_tensor.append(sample)`"""
        super().__init__(message)


class TensorDtypeMismatchError(MetaError):
    def __init__(self, expected: Union[np.dtype, str], actual: str, htype: str):
        msg = f"Dtype was expected to be '{expected}' instead it was '{actual}'. " \
              f"If you called `create_tensor` explicitly with `dtype`, your samples should also be of that dtype."

        # Sherry: we may want to raise this error at the API level to determine if the user explicitly overwrote
        #  the `dtype` or not. (to make this error message more precise)
        # Sherry: because if the user uses `dtype=np.uint8`, but the `htype` the tensor is created
        #  with has it's default dtype set as `uint8` also, then this message is ambiguous
        htype_dtype = muller.HTYPE_CONFIGURATIONS[htype].get("dtype", None)
        if htype_dtype is not None and htype_dtype == expected:
            msg += f" Htype '{htype}' expects samples to have dtype='{htype_dtype}'."
            super().__init__("")

        super().__init__(msg)


class SampleHtypeMismatchError(Exception):
    def __init__(self, htype, sample_type):
        super().__init__(
            f"htype '{htype}' does not support samples of type {sample_type}."
        )


class SampleExtendError(Exception):
    def __init__(self, message):
        message += (
            " If you wish to skip the samples that cause errors,"
            " please specify `ignore_errors=True`."
        )
        super().__init__(message)


class JsonValidationError(Exception):
    pass


class IncompatibleHtypeError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class UnsupportedCompressionError(CompressionError):
    def __init__(self, compression: Optional[str], htype: Optional[str] = None):
        if htype:
            super().__init__(
                f"Compression '{compression}' is not supported for {htype} htype."
            )
        else:
            super().__init__(
                f"Compression '{compression}' is not supported. Supported compressions: {muller.compressions}."
            )


class DatasetTooLargeToDelete(Exception):
    def __init__(self, ds_path):
        message = f"MULLER_F Dataset {ds_path} was too large to delete. Try again with large_ok=True."
        super().__init__(message)


class MergeError(Exception):
    pass


class FilterError(Exception):
    pass


class AggregateError(Exception):
    pass


class MergeMismatchError(MergeError):
    def __init__(self, tensor_name, mismatch_type, original_value, target_value):
        message = f"Unable to merge, tensor {tensor_name} has different {mismatch_type}. Current:{original_value}, " \
                  f"Target: {target_value}"
        super().__init__(message)


class MergeConflictError(MergeError):
    def __init__(self, conflict_tensors=None, message=""):
        if conflict_tensors:
            message = f"Unable to merge, tensors {conflict_tensors} have conflicts and update_resolution argument " \
                      f"was not provided. Use update_resolution='theirs' or update_resolution='ours' " \
                      f"to resolve the conflict."
            super().__init__(message)
        else:
            super().__init__(message)


class MergeLostUUid(MergeError):
    def __init__(self, target_uuid=None):
        message = f"Unable to merge, cannot find the uuid {target_uuid} in original node."
        super().__init__(message)


class TensorTooLargeToDelete(Exception):
    def __init__(self, tensor_name):
        message = f"Tensor {tensor_name} was too large to delete. Try again with large_ok=True."
        super().__init__(message)


# Sherry
class RomaGetError(Exception):
    def __init__(self):
        super().__init__()


class InvalidShapeIntervalError(Exception):
    def __init__(
            self,
            message: str,
            lower: Optional[Sequence[int]] = None,
            upper: Optional[Sequence[int]] = None,
    ):
        s = message

        if lower is not None:
            s += f" lower={str(lower)}"

        if upper is not None:
            s += f" upper={str(upper)}"

        super().__init__(s)


class SampleUpdateError(Exception):
    def __init__(self, key: str):
        super().__init__(f"Unable to update sample in tensor {key}.")


class UnAuthorizationError(Exception):
    def __init__(self, custom_message: Optional[str] = None):
        if custom_message is None:
            custom_message = "Trying to access unauthorized MULLER-F dataset in Huashan Platform! " \
                             "The uid in /tmp/user/user.info is not allowed."
        super().__init__(custom_message)


class FilterVectorizedConditionError(Exception):
    def __init__(self, condition):
        super().__init__(
            f"The filter condition ({str(condition)}) is not correct. A valid filter condition should be "
            f"(tensor_column, filter_condition, filter_value, negation), while the type of tensor_column "
            f"and filter_value should be matched and supported. E.g., (\"test\", \">\", 2, \"NOT\")."
        )


class FilterVectorizedConnectorListError(Exception):
    def __init__(self, connector_list):
        super().__init__(
            f"The connector list ({str(connector_list)}) is not correct. A valid connector list should only "
            f"contains \"AND\" or \"OR\" as elements, while len(connector_list) = len(condition_list) - 1."
        )


class InvertedIndexNotExistsError(Exception):
    def __init__(self, tensor_name):
        super().__init__(
            f"You need to create inverted index on the tensor column ({tensor_name}) using `ds.create_index(...)` "
            f"or ds.create_index_vectorized(...) (recommended!!) before this operation. "
        )


class InvertedIndexNotFoundError(Exception):
    def __init__(self, tensor_name):
        super().__init__(
            f"Cannot find all the index files for ({tensor_name}).` "
            f"You need to wait util the construction of all index files is completed. Then you can conduct the query."
        )


class InvertedIndexUnsupportedError(Exception):
    def __init__(self, operator):
        super().__init__(
            f"We currently do not support inverted_index for operator {operator}."
        )


class FilterOperatorNegationUnsupportedError(Exception):
    def __init__(self, operator):
        super().__init__(
            f"We currently do not support negation for operator {operator}."
        )


class UpdateIndexFailError(Exception):
    def __init__(self, message):
        super().__init__(message)


class MultiProcessUnsupportedError(Exception):
    def __init__(self, shape, htype, dtype):
        if shape:
            super().__init__(
                f"We currently do not support fetching data with multi processes for varies shapes and shapes with "
                f"more than 2 dimension."
            )
        else:
            super().__init__(
                f"We currently do not support fetching data with multi processes for htype {htype}, dtype {dtype}."
            )


class NumpyDataNotContinuousError(Exception):
    def __init__(self):
        super().__init__(
            f"The indices are not continuous. Please use numpy() or numpy_full()."
        )


class InvalidJsonFileName(Exception):
    def __init__(self, file_name):
        super().__init__(
            f"The file name ({file_name}) is not a valid json or jsonl file. "
        )


class InvalidNumWorkers(Exception):
    def __init__(self, num):
        super().__init__(
            f"The num of workers should be an integer larger that 0. num_workers=({num}) is invalid. "
        )


class SummaryLimit(Exception):
    def __init__(self, num, limit):
        super().__init__(
            f"The dataset has {num} samples, which is more than our limit {limit} "
            f"It might take a long time to summarize. "
            f"Please consider specifying force=True if you do not mind to wait for a long time."
        )


class ToDataFrameLimit(Exception):
    def __init__(self, num, limit):
        super().__init__(
            f"The dataset has {num} samples, which is invalid or is more than our limit {limit}. "
            f"It might take a long time to export as a dataframe. "
            f"Please consider only exporting the partial dataset by using the index_list arguments, "
            f"or directly specifying force=True if you do not mind to wait for a long time."
        )


class ExportDataFrameLimit(Exception):
    def __init__(self, num, limit):
        super().__init__(
            f"The dataset has {num} samples, which is more than our limit {limit}. "
            f"It might take longer than 5s to export as a pandas dataframe. "
            f"Please directly specifying force=True if you do not mind to wait for a while."
        )


class InvalidTensorList(Exception):
    def __init__(self, tensor_list):
        super().__init__(
            f"The argument {str(tensor_list)} is invalid. Please check whether each element provided in the list is "
            f"the a valid tensor in the dataset."
        )


class MetaNotFound(Exception):
    def __init__(self, meta, meta_type):
        super().__init__(f"{meta} not found or is not the expected type {meta_type}.")


class TransformFailure(Exception):
    def __init__(self):
        super().__init__(f"Transform fails.")
