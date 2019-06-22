(* ::Package:: *)

(* :Title: HDF5Tools *)

(* :Authors: Stewart Dickson, Rafal Chojna *)

(* :Summary:
This package organizes and provides usage help for the LibraryLink wrapping
of the HDF5 API.
*)

(* :Context: HDF5Tools` *)

(* :Package Version: 2.0 *)

(* :History:
		  V.1.0 Apr 20 2012, by Stewart Dickson
		  V.2.0 Oct 1, 2016, by Rafal Chojna
*)

(* :Copyright: Copyright 2016, Wolfram Research, Inc. *)

(* :Keywords:
	Hierarchical, Data, Format
*)

(* :Sources:
The HDF Group, "HDF5: API SPecification Reference Manual",
http://www.hdfgroup.org/HDF5/doc/RM/RM_H5Front.html
*)

(* :Mathematica Version: 11.1 *)

BeginPackage["HDF5Tools`"]

(* ::Section::Closed:: *)
(* Usage tags *)

HDF5ToolsInit::usage = "HDF5ToolsInit[] Loads library and low-level wrappers.";

HDF5ishdf5::usage = "HDF5ishdf5[fileName_String] Returns True if the fileName names an HDF5 file.";

HDF5OpenFile::usage = "HDF5openFile[fileName_String,mode_Integer] Opens an HDF5 file by name and returns the opened file handle";

HDF5FileHandle::usage = "HDF5FileHandle[fileId_Integer] Open HDF5 file object";

HDF5CloseFile::usage = "HDF5CloseFile[HDF5FileHandle[fileId_Integer] Closes an HDF5 file previously opened using HDF5OpenFile.";

HDF5ObjectHandle::usage = "HDF5ObjectHandle[objectId_Integer] Open HDF5 generic object of unknown type";

HDF5GroupHandle::usage = "";

HDF5Group::usage = "HDF5Group[name_String][List] -- Represents the hierarchy of an HDF5 Group object";

HDF5DataspaceHandle::usage = "HDF5DataspaceHandle[dspaceID] Represents an open dataspace with given ID";

HDF5CreateDataspace::usage = "HDF5CreateDataspace[dims_List, maxDims_List] Creates a simple dataspace with current dims equal to dims and extendable to maxDims.
HDF5CreateDataspace[dims_List] Creates simple dataspace of fixed size.
HDF5CreateDataspace[0 | {}, _] Creates a scalar dataspace.";

HDF5GetDataspaceDimensions::usage = "HDF5GetDataspaceDimensions[HDF5DataspaceHandle[dspaceID_]] Returns a list of dimensions of the dataspace.";

HDF5CloseDataspace::usage = "HDF5CloseDataspace[HDF5DataspaceHandle[dspaceId]] Closes a Dataset previously opened using HDF5DatasetGroup.";

HDF5DatasetHandle::usage = "";

HDF5DatatypeHandle::usage = "";

HDF5PropListHandle::usage = "HDF5PropListHandle[plistID] Represents an open property list with given ID";

HDF5CreateDataset::usage = "HDF5CreateDataset[HDF5FileHandle[id_], path_String, dtypeID_Integer, dspaceID_Integer, lcPropList_Integer, dsetCreatPropList_Integer] 
	Creates a dataset with given absolute path, datatype, dataspace, link creation property list and dataset creation property list."

HDF5OpenDataset::usage = "HDF5OpenDataset[ HDF5FileHandle[ fileId_], path_String] Opens a named Dataset in an HDF5 file and returns the opened Dataset handle";

HDF5DatasetTypeString::usage = "HDF5DatasetTypeString[ HDF5DatasetHandle[datasetId_]]";

HDF5DatasetDims::usage = "HDF5DatasetDims[ HDF5DatasetHandle[datasetId_]]";

HDF5DatasetEncoding::usage = "HDF5DatasetEncoding[ HDF5DatasetHandle[datasetId_]]";

HDF5ReadDatasetRaw::usage = "HDF5ReadDatasetRaw[ HDF5DatasetHandle[datasetId_], opt___] Reads contents of the dataset. Does not normalize NumericArrays.";

HDF5ReadDataset::usage = "HDF5ReadDataset[ HDF5DatasetHandle[datasetId_], opt___] Reads contents of the dataset. Normalizes NumericArrays.";

HDF5CloseDataset::usage = "HDF5CloseDataset[ HDF5DatasetHandle[datasetId_]]";

HDF5OpenGroup::usage = "HDF5OpenGroup[ HDF5FileHandle[ fileId_], groupName_String] Opens a named Group in an HDF5 file and returns the opened group handle.";

HDF5CreateGroup::usage =
    "HDF5CreateGroup[HDF5FileHandle[id_], path_, HDF5PropListHandle[listID_Integer]] Creates a group with given absolute path and specified link property list.
HDF5CreateGroup[HDF5FileHandle[id_], path_] Creates a group with given absolute path and default link property list (UTF8 allowed and create intermediate groups).";

HDF5CloseGroup::usage = "HDF5CloseGroup[ HDF5GroupHandle[groupId_] ] Closes a Group previously opened using HDF5OpenGroup.";

HDF5CreateDatatype::usage = "HDF5CreateDatatype[dtypeClass_Integer, size_Integer] Creates a new datatype of given class (H5TCOMPOUND, H5TOPAQUE, H5TENUM or H5TSTRING) and size.";

HDF5OpenDatatype::usage = "HDF5OpenDatatype[ HDF5FileHandle[ fileId_], path_String] Opens a named Datatype in an HDF5 file and returns the opened Datatype handle";

HDF5CloseDatatype::usage = "HDF5CloseDatatype[ HDF5DatatypeHandle[dtypeId_] ] Closes a Datatype previously opened using HDF5OpenDatatype.";

HDF5CopyDatatype::usage = "HDF5CopyDatatype[HDF5DatatypeHandle[dtypeId_]] Creates a modifiable copy of given datatype.";

HDF5CommitDatatype::usage =
    "HDF5CommitDatatype[HDF5FileHandle[fid_], path_String, HDF5DatatypeHandle[dtypeID_], HDF5PropListHandle[listID_Integer]] Commits a datatype under given path in the file.
HDF5CommitDatatype[HDF5FileHandle[fid_], path_String, HDF5DatatypeHandle[dtypeID_]] Commits a datatype under given path in the file.";

HDF5CommitTypeFromString::usage = "HDF5CommitTypeFromString[fid_HDF5FileHandle, dtypePath_String, typeDescription_String] Creates new committed datatype from the text description.";

HDF5CreateTypeFromString::usage = "HDF5CreateTypeFromString[typeDescription_String] Creates new datatype from text description and returns its id.";

HDF5CreatePropertyList::usage = "HDF5CreatePropertyList[class_Integer] Creates property list of given class (for example H5PFILECREATE, H5PLINKACCESS, etc)";

HDF5ClosePropertyList::usage = "HDF5ClosePropertyList[HDF5PropListHandle[listID_Integer]] Closes previously opened property list.";

HDF5ObjectType::usage = "HDF5ObjectType[ HDF5GroupHandle[ gid_], linkName_String] Return the type of object pointed to by the named link";

HDF5OpenObject::usage = "HDF5OpenObject[ HDF5FileHandle[ fileId_], objectName_String] Opens a named object of unknown type in an HDF5 file and returns the opened object handle.";

HDF5CloseObject::usage = "HDF5CloseObject[ HDF5ObjectHandle[objectId_] ] Closes an object previously opened using HDF5OpenObject.";

HDF5DeleteLink::usage = "HDF5DeleteLink[HDF5GroupHandle[gid_], linkName_String] Deletes a link. Use with caution as it may cause irreversible data loss.";

HDF5LinkExistsQ::usage = "HDF5LinkExistsQ[HDF5GroupHandle[gid_], linkName_String] Returns True iff the link with given name exists in the group.";

HDF5CreateSoftLink::usage =
    "HDF5CreateSoftLink[HDF5GroupHandle[gid_], linkName_String, targetPath_String, HDF5PropListHandle[listID_Integer]] Creates a new soft link named linkName, located in group gid and pointing to
    targetPath.";

HDF5CreateHardLink::usage =
    "HDF5CreateHardLink[HDF5GroupHandle[gid_], linkName_String, targetPath_String, HDF5PropListHandle[listID_Integer]] Creates a new hard link named linkName, located in group gid and pointing to
    targetPath.";

HDF5MissingObject::usage = "HDF5MissingObject[name_String] -- Place holder to represent links to objects that do not exist";

HDF5InaccessibleObject::usage = "HDF5InaccessibleObject[name_String] -- Place holder to represent an HDF5 object that could not be opened";

HDF5UnknownObject::usage = "HDF5UnknownObject[name_String] -- Place holder to represent an HDF5 object of unknown type";

HDF5GetFilterString::usage = "HDF5GetFilterString[filterId_] Returns the filter - usually compression - applied to a dataset";

HDF5GetGroupContents::usage = "HDF5GetGroupContents[ HDF5FileHandle[ fileId_], groupName_String]";

HDF5GetStructuredDatasets::usage = "HDF5GetStructuredDatasets[ HDF5FileHandle[fileId_], groupName_String, lvl_]";

HDF5GetAttributes::usage = "HDF5GetAttributes[fileName_String]";

HDF5GetGroupNames::usage = "HDF5GetGroupNames[fileName_String]";

HDF5GetDatasetNames::usage = "HDF5GetDatasetNames[fileName_String]";

HDF5GetSummary::usage = "HDF5GetSummary[fileName_String] Prints summary of given HDF5 file."

HDF5CreateCompound::usage = "HDF5CreateCompound[<|name1->type1, name2->type2, ... |>] Creates compound datatype with members of specified names and types.";

HDF5CreateEnum::usage = "HDF5CreateEnum[baseType, <|name1->value1, name2->value2, ... |>] Creates enumerated datatype with specified names and corresponding values.";

HDF5PrintType::usage = "HDF5PrintType[typeId_Integer, verbosity_] Prints information about given datatype."

HDF5RegisterComplexType::usage = "HDF5RegisterComplexType[None | Automatic | {re_String, im_String}] - registers new type to be treated as complex numbers";

HDF5UnregisterComplexType::usage = "HDF5RegisterComplexType[] - unregisters the type registered for complex numbers";

HDF5StructureGraph::usage = "HDF5StructureGraph[ HDF5FileHandle[fileId_] ] Draws a graph representing the structure of given HDF5 file." 

HDF5StructureGraphLegend::usage = "Returns a legend for HDF5StructureGraph."

HDF5AttributeHandle::usage = "";

HDF5CreateAttribute::usage = "HDF5CreateAttribute[HDF5ObjectHandle[id_], attrName_String, dtypeID_Integer, dspaceID_Integer, lcPropList_Integer]
Creates a new attribute with given name attached to given object."

HDF5GetAttributeDataspace::usage = "HDF5GetAttributeDataspace[HDF5AttributeHandle[attrID_]] Returns a dataspace handle to the Attribute's data space.";

HDF5AttributeExists::usage = "HDF5AttributeExists[o_HDF5ObjectHandle, attrName_String] Returns True iff the attribute with given name is attached to the given object.";

HDF5DeleteAttribute::usage = "HDF5DeleteAttribute[o_HDF5ObjectHandle, attrName_String] Deletes a named attribute.";

h5acreate::usage = "h5acreate[objId_Integer, attrName_String, typeId_Integer, spaceId_Integer, acplId_Integer, aaplId_Integer] Returns the Id of the newly created attribute attached to the given object.";

h5aexists::usage = "h5aexists[objId_Integer, attrName_String] Returns a positive value if and only if the attribute with given name is attached to the given object.";

h5aopen::usage = "h5aopen[objId_Integer, attrName_String] Returns the Id of a named attribute attached to the given object.";

h5adelete::usage = "h5adelete[objId_Integer, attrName_String] Deletes a named attribute attached to the given object.";

h5agetnamebyidx::usage = "h5agetnamebyidx[objId_Integer, objName_String, idxField_Integer, order_Integer, index_Integer] Retrieves name of the nth attribute of an open HDF5 file object";

h5agetspace::usage = "h5agetspace[attrId_Integer] Returns the Id of the data storage space of the attribute";

h5agettype::usage = "h5agettype[attrId_Integer] Returns the identifier of an attribute's value type.";

h5areadnumericarray::usage = "h5areadnumericarray[attrId_Integer, memtypeId_Integer] Read an attribute from an HDF5 to NumericArray object";

h5areadstrings::usage = "h5areadstrings[attrId_Integer, memtypeId_Integer] Read an attribute from an HDF5 file which type is String";

h5areadcompounds::usage = "h5areadcompounds[attrId_Integer, memtypeId_Integer] Read an attribute from an HDF5 file where type is compound";

h5areadrawbytes::usage = "h5areadrawbytes[attrId_Integer, memtypeId_Integer] Read an attribute from an HDF5 file with raw bytes.";

h5areadarrays::usage = "h5areadarrays[attrId_Integer, memtypeId_Integer] Read an attribute from an HDF5 file where type is Array";

h5areadtensorint::usage = "h5areadtensorint[attrId_Integer] Read integer data from attribute to MTensor.";

h5areadtensorreal::usage = "h5areadtensorreal[attrId_Integer] Read real data from attribute to MTensor.";

h5areadtensorcomplex::usage = "h5areadtensorcomplex[attrId_Integer] Read complex data from attribute to MTensor.";

h5awritearray::usage = "h5awritearray[attrId_Integer, memTypeId_Integer, attrValue_List] Writes array data to an attribute.";

h5awritecompound::usage = "h5awritecompound[attrId_Integer, memTypeId_Integer, attrValue_List] Writes compound data to an attribute.";

h5awritecomplex::usage = "h5awritecomplex[attrId_Integer, memTypeId_Integer, attrValue_List] Writes complex data to an attribute.";

h5awriteinteger::usage = "h5awriteinteger[attrId_Integer, memTypeId_Integer, attrValue_List] Writes integer data to an attribute.";

h5awritereal::usage = "h5awritereal[attrId_Integer, memTypeId_Integer, attrValue_List] Writes real data to an attribute.";

h5awritestring::usage = "h5awritestring[attrId_Integer, memTypeId_Integer, attrValue_List] Writes string data to an attribute.";

h5awritenumericarray::usage = "h5awritenumericarray[attrId_Integer, memtypeId_Integer, attrValue_List] Write numeric data to an attribute.";

h5awriterawbyte::usage = "h5awriterawbyte[attrId_Integer, memtypeId_Integer, attrValue_List] Write data as raw bytes to an attribute.";

h5aclose::usage = "h5aclose[attrId_Integer] Releases the resources associated with an opened attribute";

h5dcreate::usage = "h5dcreate[locId_Integer, dsetName_String, dtypeId_Integer, dspaceId_Integer, lcplId_Integer, dcplId_Integer, daplId_Integer] Returns the Id of the newly created dataset in the HDF5 file.";

h5dopen::usage = "h5dopen[locId_Integer, dsetName_String, daplId_Integer] Returns the Id of the opened dataset in the HDF5 file.";

h5dgetspace::usage = "h5dgetspace[dsetId_Integer] Returns the Id of the dataspace extracted from the opened dataset in the HDF5 file.";

h5dgetstoragesize::usage = "h5dgetstoragesize[dsetId_Integer] Returns the amount of storage space, in bytes, allocated for the dataset, not counting metadata.";

h5dgettype::usage = "h5dgettype[dsetId_Integer] Returns the Id of the datatype extracted from the opened dataset in the HDF5 file.";

h5dgetcreateplist::usage = "h5dgetcreateplist[dsetId_Integer] Returns an identifier for a copy of the dataset creation property list for a dataset.";

h5dreadstrings::usage = "h5dreadstrings[dsetId_Integer, memtypeId_Integer, memspaceId_Integer, dspaceID_Integer, xferPlistId_Integer] Read a dataset from an HDF5 file where type is String";

h5dreadarrays::usage = "h5dreadarrays[dsetId_Integer, memtypeId_Integer, memspaceId_Integer, dspaceID_Integer, xferPlistId_Integer] Read a dataset from an HDF5 file where type is Array";

h5dreadcompounds::usage = "h5dreadcompounds[dsetId_Integer, memtypeId_Integer, memspaceId_Integer, dspaceID_Integer, xferPlistId_Integer] Read a dataset from an HDF5 file where type is compound";

h5dreadrawbytes::usage = "h5dreadrawbytes[dsetId_Integer, memtypeId_Integer, memspaceId_Integer, dspaceID_Integer, xferPlistId_Integer] Read a dataset from an HDF5 file with raw bytes.";

h5dreadnumericarray::usage = "h5dreadnumericarray[dsetId_Integer, memtypeId_Integer, memspaceId_Integer, dspaceID_Integer, xferPlistId_Integer] Read a dataset from an HDF5 file to NumericArray";

h5dreadnumericarrayinto::usage = "h5dreadnumericarrayinto[dsetId_Integer, memtypeId_Integer, memspaceId_Integer, dspaceID_Integer, xferPlistId_Integer, buffer_NumericArray] "<>
	"Read a dataset from an HDF5 file to already existing NumericArray. The size and type of NumericArray will not be verified, so use with caution.";

h5dreadtensorint::usage = "h5dreadtensorint[dsetId_Integer, memspaceId_Integer, dspaceID_Integer, xferPlistId_Integer] Read integer data from dataset to MTensor.";

h5dreadtensorreal::usage = "h5dreadtensorreal[dsetId_Integer, memspaceId_Integer, dspaceID_Integer, xferPlistId_Integer] Read real data from dataset to MTensor.";

h5dreadtensorcomplex::usage = "h5dreadtensorcomplex[dsetId_Integer, memspaceId_Integer, dspaceID_Integer, xferPlistId_Integer] Read complex data from dataset to MTensor.";

h5dsetextent::usage = "h5dsetextent[dssetId_Integer, size_List] Changes the sizes of a dataset's dimensions.";

h5dwriteinteger::usage = "h5dwriteinteger[dsetId_Integer, memtypeId_Integer, memspaceId_Integer, dspaceID_Integer, xferPlistId_Integer, data_List] Write a dataset to an HDF5 file where type is Integer";

h5dwritereal::usage = "h5dwritereal[dsetId_Integer, memtypeId_Integer, memspaceId_Integer, dspaceID_Integer, xferPlistId_Integer, data_List] Write a dataset to an HDF5 file where type is Real";

h5dwritecomplex::usage = "h5dwritecomplex[dsetId_Integer, memtypeId_Integer, memspaceId_Integer, dspaceID_Integer, xferPlistId_Integer, data_List] Write a dataset to an HDF5 file where type is Complex";

h5dwritestring::usage = "h5dwritestring[dsetId_Integer, memtypeId_Integer, memspaceId_Integer, dspaceId_Integer, xferPlistId_Integer, data_List] Write a dataset to an HDF5 file where type is String";

h5dwritearray::usage = "h5dwritearray[dsetId_Integer, memtypeId_Integer, memspaceId_Integer, dspaceId_Integer, xferPlistId_Integer, data_List] Write a dataset to an HDF5 file where type is Array";

h5dwritecompound::usage = "h5dwritecompound[dsetId_Integer, memtypeId_Integer, memspaceId_Integer, dspaceId_Integer, xferPlistId_Integer, data_List] Write a dataset to an HDF5 file where type is Compound";

h5dwritenumericarray::usage = "h5dwritenumericarray[dsetId_Integer, memtypeId_Integer, memspaceId_Integer, dspaceID_Integer, xferPlistId_Integer, data_List] Write a numeric dataset to an HDF5 file.";

h5dwriterawbyte::usage = "h5dwriterawbyte[dsetId_Integer, memtypeId_Integer, memspaceId_Integer, dspaceID_Integer, xferPlistId_Integer, data_List] Write a dataset of raw bytes to an HDF5 file.";

h5dclose::usage = "h5dclose[dssetId_Integer] Terminates access to a previously opened dataset of an HDF5 file.";

h5esetfile::usage = "h5esetfile[filePath_String] Creates a file for temporary storage of error messages.";

h5egeterrmsg::usage = "h5egeterrmsg[] Returns all error messages thrown since the last call to this function.";

h5fishdf5::usage = "h5fishdf5[fileName_String] Returns True if the fileName indicates an HDF5 file.";

h5fcreate::usage = "h5fcreate[fileName_String, flags_Integer] Creates an HDF5 file on disk and returns the opened file descriptor.";

h5fflush::usage = "h5fflush[objectId_Integer, scope_Integer] Flushes all buffers associated with a file to disk. ";

h5fgetfilesize::usage = "h5fgetfilesize[fileId_Integer] Returns total size of currently-open file (negative value if error).";

h5fopen::usage = "h5fopen[fileName_String, mode_Integer] Opens an HDF5 file by name and returns the opened file descriptor.";

h5fclose::usage = "h5fclose[fileId_Integer] Terminates access to a previously opened HDF5 file.";

h5gcreate::usage = "h5gcreate[locId_Integer, groupName_String, lcplId_Integer, gcplId_Integer, gaplId_Integer] Creates a new empty group and links it to a location in the file.";

h5gopen::usage = "h5gopen[locId_Integer, groupName_String] Opens a named Group in an HDF5 file and returns the opened group identifier.";

h5ggetinfonlinks::usage = "h5ggetinfonlinks[groupId_Integer] Returns the number of links in a Group.";

h5gclose::usage = "h5gclose[groupId_Integer] Terminates access to a previously opened HDF5 file Group.";

h5lcreatehard::usage = "h5lcreatehard[objLocId_Integer, objName_String, linkLocId_Integer, linkName_String, lcplId_Integer, laplId_Integer] Creates a hard link to an object.";

h5lcreatesoft::usage = "h5lcreatesoft[targetPath_String, linkLocId_Integer, linkName_String, lcplId_Integer, laplId_Integer] Creates a soft link to an object.";

h5lexists::usage = "h5lexists[locId_Integer, linkName_String, laplId_Integer] Returns TRUE or FALSE.";

h5lgetinfotype::usage = "h5lgetinfotype[locId_Integer, linkName_String, laplId_Integer] Returns the enumerated type of the link.";

h5lgetnamebyidx::usage = "h5lgetnamebyidx[locId_Integer, groupName_String, idxField_Integer, order_Integer, index_Integer] Retrieves name of the nth link in an open HDF5 file group";

h5ltpathvalid::usage = "h5ltpathvalid[locId_Integer, path_String, checkObjectQ_] Determines whether an HDF5 path is valid and, optionally, whether the path resolves to an HDF5 object.";

h5lttexttodtype::usage = "h5lttexttodtype[dtypeDescription_String] Creates a new datatype from a textual description.";

h5ltdtypetotext::usage = "h5ltdtypetotext[dtypeId_Integer] Returns a textual description of given datatype.";

h5oexistsbyname::usage = "h5oexistsbyname[locId_Integer, objectName_String, laplId_Integer] Returns TRUE (1) or FALSE (0) if successful; otherwise returns a negative value.";

h5oopen::usage = "h5oopen[locId_Integer, objectName_String, laplId_Integer] returns an identifier of an object opened within an open HDF5 file.";

h5ogetinfoaddr::usage = "h5ogetinfoaddr[objectId_Integer] returns the unique address of the object in the HDF5 file.";

h5ogetinfonumattrs::usage = "h5ogetinfonumattrs[objectId_Integer] returns the number of attributes attached to the object";

h5ogetinfotype::usage = "h5ogetinfotype[objectId_Integer] returns the type of the indicated object.";

h5ogetinfobynameaddr::usage = "h5ogetinfobynameaddr[locId_Integer, objName_String] returns the unique address of the object objName in the locId location.";

h5oclose::usage = "h5oclose[object_ID] releases the resources associated with a previously opened object in an HDF5 file.";

h5pgetclassid::usage = "h5pgetclassid[mmaClassId_Integer] Get the actual class id corresponding to the Mathematica class alias.";

h5pcreate::usage = "h5pcreate[classId_Integer] Creates property list of the given class.";

h5pgetlayout::usage = "h5pgetlayout[plistId_Integer] returns the layout of the raw data for a dataset. This function is only valid for dataset creation property lists.";

h5pgetchunk::usage = "h5pgetchunk[plistId_Integer, rank_Integer] retrieves the size of chunks for the raw data of a chunked layout dataset. Only valid for dataset creation property lists.";

h5pgetfilterid::usage = "h5pgetfilterid[plistId_Integer, n_Integer] retrieves the unique id of n-th filter in a filter pipeline in property list plistId.";

h5pgetnfilters::usage = "h5pgetnfilters[plistId_Integer] Returns the number of filters in the pipeline.";

h5psetcharencoding::usage = "h5psetcharencoding[plistID_Integer, encoding_Integer] Sets the character encoding used to encode link and attribute names.";

h5psetlayout::usage = "h5psetlayout[plistID_Integer, layout_Integer]  sets the type of storage used to store the raw data for a dataset.
	This function is only valid for dataset creation property lists.";

h5psetchunk::usage = "h5psetchunk[plistId_Integer, rank_Integer, dims_List] Sets chunk size (in dataset elements) for the given property list of class H5PDATASETCREATE.";

h5psetcreateintermediategroup::usage = "h5psetcreateintermediategroup[plistID_integer, crtIntGr_Integer] Specifies in property list whether to create missing intermediate groups.";

h5psetdeflate::usage = "h5psetdeflate[plistId_Integer, level_Integer] Sets deflate (GNU gzip) compression method and compression level.";

h5psetfletcher32::usage = "h5psetfletcher32[plistId_Integer] Sets up use of the Fletcher32 checksum filter.";

h5psetscaleoffset::usage = "h5psetscaleoffset[plistId_Integer, scaleType_Integer, scaleFactor_List] Sets up the use of the scale-offset filter.";

h5psetshuffle::usage = "h5psetshuffle[plistId_Integer] Sets up use of the shuffle filter.";

h5psetszip::usage = "h5psetszip[plistId_Integer, optionMask_Integer, pixelsPerBlock_List] Sets up use of the SZIP compression filter. See HDF5 docs for more details.";

h5pclose::usage = "h5pclose[plist_ID] releases the resources associated with a previously opened property list.";

h5screate::usage = "h5screate[spaceTypeId_Integer] Returns the Id of a newly created dataspace of given type.";

h5scopy::usage = "h5scopy[dspaceId_Integer] Returns the Id of a dataspace duplicated from the given dataspace Id.";

h5screatesimple::usage = "h5screatesimple[rank_Integer, currentDims_List, maxDims_List] Create a tensor dataspace in memory for reading into from a file";

h5screatesimplen::usage = "h5screatesimplen[rank_Integer, currentDims_List] Wrapper of H5Screate_simple(rank, currentDims, NULL).";

h5sgetsimpleextentdims::usage = "h5sgetsimpleextentdims[dspaceId_Integer] Returns the list of dimensions of the previously opened dataspace in an HDF5 file.";

h5sgetsimpleextentmaxdims::usage = "h5sgetsimpleextentmaxdims[dspaceId_Integer] Returns the list of maximal dimensions of the previously opened dataspace in an HDF5 file.";

h5sgetsimpleextentndims::usage = "h5sgetsimpleextentndims[dspaceId_Integer] Returns the dimensionality or rank of the previously opened dataspace in an HDF5 file.";

h5sgetsimpleextentnpoints::usage = "h5sgetsimpleextentnpoints[spaceId_Integer] Returns the number of elements in the previously opened dataspace in an HDF5 file.";

h5sgetsimpleextenttype::usage = "h5sgetsimpleextenttype[dspaceId_Integer] Returns the class Id of the data type of the data space.";

h5sselecthyperslab::usage = "h5sselecthyperslab[dspaceId_Integer, selectOp_Integer, start_List, stride_List, count_List, block_List] Set a subset selection of a dataspace";

h5sselectelements::usage = "h5sselectelements[dspaceId_Integer, selectOp_Integer, numElems_Integer, coords_NumericArray[\"UnsignedInteger64\"]] Select specified points in dataspace.";

h5sclose::usage = "h5sclose[dspaceId_Integer] Terminates access to a previously opened dataspace of an HDF5 file dataset.";

h5tgettype::usage = "h5tgettype[mmaTypeId_Integer] Get the actual type id corresponding to the Mathematica type alias.";

h5tconvert::usage = "h5tconvert[typeClass_Integer, typeSize_Integer] Converts data from one specified datatype to another.";

h5tcreate::usage = "h5tcreate[typeClass_Integer, typeSize_Integer] Return the Id of the newly created, user-defined Datatype.";

h5tenumcreate::usage = "h5tenumcreate[baseTypeId_Integer] Return the Id of the newly created enumerated datatype.";

h5tcreatestr::usage = "h5tcreatestr[strType_Integer, strCset_Integer, strSize_Integer, strPad_Integer] Return the Id of the newly created string type.";

h5tarraycreate::usage = "h5tarraycreate[baseTypeId_Integer, rank_Integer, dims_List] Return the Id of the newly created array type.";

h5tequal::usage = "h5tequal[typeId1_Integer, typeId2_Integer] Check whether two types are equal.";

h5tinsert::usage = "h5tinsert[dtypeID_Integer, dtypeName_String, offset_Integer, fieldId_Integer] Insert a field into a user-defined COMPOUND Datatype.";

h5tenuminsert::usage = "h5tenuminsert[eTypeID_Integer, fieldName_String, value_Integer] Insert named field with given value into a user-defined enumerated Datatype.";

h5tcommit::usage = "h5tcommit[locId_Integer, dtypeName_String, dtypeId_Integer, lcplId_Integer, tcplId_Integer, taplId_Integer ] Commits a user-defined Datatype to the HDF5 file.";

h5tcopy::usage = "h5tcopy[dtypeId_Integer] copies an existing datatype.";

h5topen::usage = "h5topen[locId_Integer, name_String, taplId_Integer] Return the datatype Id of the Named Datatype in the Group locId.";

h5tgetclass::usage = "h5tgetclass[dtypeId_Integer] Returns the enum value for the data type class";

h5tgetnativetype::usage = "h5tgetnativetype[dtypeId_Integer, direction_Integer] Get the machine native datatype corresponding to the file datatype.";

h5tgetarrayndims::usage = "h5tgetarrayndims[adtypeId_Integer] Returns the number of dimensions in an Array data type.";

h5tgetarraydims::usage = "h5tgetarraydims[adtypeId_Integer] Returns a flat list of dimensions in an Array data type.";

h5tgetnmembers::usage = "h5tgetnmembers[dtypeId_Integer] Returns the number of members in a Compound or Enumeration data type.";

h5tgetmembername::usage = "h5tgetmembername[dtypeId_Integer, fieldIndex_Integer] Retrieves the name of a compound or enumeration datatype member.";

h5tgetmemberoffset::usage = "h5tgetmemberoffset[dtypeId_Integer, fieldIndex_Integer] Retrieves the offset of a compound or enumeration datatype member.";

h5tgetmembertype::usage = "h5tgetmembertype[dtypeId_Integer, fieldIndex_Integer] returns the datatype of a member of a compound or enumeration.";

h5tgetmembervalue::usage = "h5tgetmembervalue[dtypeId_Integer, fieldIndex_Integer] returns the value of an integer enumeration datatype member.";

h5tgetnamestring::usage = "h5tgetnamestring[dtypeId_Integer] Returns a String describing a datatype";

h5tgettag::usage = "h5tgettag[dtypeId_Integer] Gets the tag associated with an opaque datatype.";

h5tsettag::usage = "h5tsettag[dtypeId_Integer, tag_String] Tags an opaque datatype.";

h5tgetsize::usage = "h5tgetsize[dtypeId_Integer] Returns the size of the data type";

h5tgetsuper::usage = "h5tgetsuper[dtypeId_Integer] Returns the base datatype from which the datatype dtypeId is derived. ";

h5tclose::usage = "h5tclose[dspaceId_Integer] Terminates access to a previously opened datatype of an HDF5 file dataset.";

h5zfilteravail::usage = "h5zfilteravail[filter_Integer] Determines whether a filter is available.";

h5zgetfilterinfo::usage = "h5zgetfilterinfo[filter_Integer] Returns two values determining whether a filter is encode and decode enabled.";

h5aread::usage = "h5aread[attrId_Integer, memtypeId_Integer] Read an attribute from an HDF5 file";

h5awrite::usage = "h5awrite[attrId_Integer, memtypeId_Integer, attrValue_List] Write data to an attribute.";

h5dread::usage = "h5dread[dsetId_Integer, memtypeId_Integer, memspaceId_Integer, dspaceID_Integer, xferPlistId_Integer] Read a dataset from an HDF5 file.";

h5dwrite::usage = "h5dwrite[dsetId_Integer, memtypeId_Integer, memspaceId_Integer, dspaceID_Integer, xferPlistId_Integer, data_] Write a dataset to an HDF5 file. Data can be a hyper-rectangular List or NumericArray";
	
(* ::Section::Closed:: *)
(* HDF5 Constants (defined in $HDF5/src/H5*public.h header files) *)
	
(* Defines from $HDF5/src/H5public.h *)
H5INDEXUNKNOWN 	= (-1);
H5INDEXNAME 	= 0;
H5INDEXCRTORDER = 1;

H5ITERUNKNOWN 	= (-1);
H5ITERINC 		= 0;
H5ITERDEC	 	= 1;
H5ITERNATIVE 	= 2;

H5PDEFAULT = 0;

(* Defines from $HDF5/src/H5Dpublic.h *)
H5DLAYOUTERROR 	= (-1);
H5DCOMPACT 		= 0;
H5DCONTIGUOUS 	= 1;
H5DCHUNKED 		= 2;
H5DNLAYOUTS 	= 3;

(* Defines from $HDF5/src/H5Fpublic.h *)
H5FACCRDONLY 	= 0;
H5FACCRDWR 		= 1;
H5FACCTRUNC 	= 2;   		(* overwrite existing files								*)
H5FACCEXCL 		= 4;  		(* fail if file already exists							*)
H5FACCDEBUG		= 8;  		(* print debug info 									*)
H5FACCCREAT 	= 16; 		(* create non-existing files 							*)
H5FACCDEFAULT 	= 65535; 	(* ignore setting on lapl (Link Access Property List)	*)

H5FSCOPELOCAL 	= 0;	(* specified file handle only	*)
H5FSCOPEGLOBAL 	= 1; 	(* entire virtual file			*)

(* Defines from $HDF5/src/H5Lpublic.h *)
H5LSAMELOC 		= 0;	  (* Macro to indicate operation occurs on same location *)

H5LTYPEERROR 	= (-1);   (* Invalid link type id *)
H5LTYPEHARD 	= 0;	  (* Hard link id		  *)
H5LTYPESOFT 	= 1;	  (* Soft link id		  *)
H5LTYPEEXTERNAL = 64;	  (* External link id	  *)
H5LTYPEMAX 		= 255;	  (* Maximum link type id *)

(* Defines from $HDF5/src/H5Opublic.h *)
H5OTYPEUNKNOWN	   		= (-1);	(* Unknown object type			 	*)
H5OTYPEGROUP		 	=  0;	(* Object is a group				*)
H5OTYPEDATASET	   		=  1;	(* Object is a dataset			  	*)
H5OTYPENAMEDDATATYPE 	=  2;	(* Object is a named data type	  	*)
H5OTYPENTYPES			=  3;	(* Number of different object types *)

(* Constants from $HDF5/src/H5Ppublic.h *)
h5pbuiltinclasses = {
	H5PROOT,
	H5POBJECTCREATE,
	H5PFILECREATE,
	H5PFILEACCESS,
	H5PDATASETCREATE,
	H5PDATASETACCESS,
	H5PDATASETXFER,
	H5PFILEMOUNT,
	H5PGROUPCREATE,
	H5PGROUPACCESS,
	H5PDATATYPECREATE,
	H5PDATATYPEACCESS,
	H5PSTRINGCREATE,
	H5PATTRIBUTECREATE,
	H5POBJECTCOPY,
	H5PLINKCREATE,
	H5PLINKACCESS
};
		
(* Defines from $HDF5/src/H5Spublic.h *)
H5SALL			=  0;  	(* Default DataSpace ID *)
H5SUNLIMITED 	= (-1);
H5SMAXRANK   	= 32; 	(* user-level maximum number of dimensions *)

H5SNOCLASS  = (-1); 	(* error                       *)
H5SSCALAR   =  0;  		(* scalar variable             *)
H5SSIMPLE   =  1;  		(* simple data space (MTensor) *)
H5SNULL     =  2;

H5SSELECTNOOP   = (-1); (* error                                     *)
H5SSELECTSET    =  0; (* Select "set" operation                    *)
H5SSELECTOR     =  1; (* Binary "or" operation for hyperslabs (add new selection to existing selection)
               * Original region:  AAAAAAAAAA
               * New region:             BBBBBBBBBB
               * A or B:           CCCCCCCCCCCCCCCC
               *)
H5SSELECTAND    =  2; (* Binary "and" operation for hyperslabs (only leave overlapped regions in selection)
               * Original region:  AAAAAAAAAA
               * New region:             BBBBBBBBBB
               * A and B:                CCCC              *)
H5SSELECTXOR    =  3; (* Binary "xor" operation for hyperslabs (only leave non-overlapped regions in selection)
               * Original region:  AAAAAAAAAA
               * New region:             BBBBBBBBBB
               * A xor B:          CCCCCC    CCCCCC        *)
H5SSELECTNOTB   =  4;  (* Binary "not" operation for hyperslabs (only leave non-overlapped regions in original selection)
                * Original region:  AAAAAAAAAA
                * New region:             BBBBBBBBBB
                * A not B:          CCCCCC
                *)
H5SSELECTNOTA   =  5;  (* Binary "not" operation for hyperslabs (only leave non-overlapped regions in new selection)
                * Original region:  AAAAAAAAAA
                * New region:             BBBBBBBBBB
                * B not A:                    CCCCCC
                *)
H5SSELECTAPPEND  =  6; (* Append elements to end of point selection *)
H5SSELECTPREPEND =  7; (* Prepend elements to beginning of point selection *)
H5SSELECTINVALID =  8; (* Invalid upper bound on selection operations *)

	
(* Defines from $HDF5/src/H5Tpublic.h *)
H5TNOCLASS         = (-1);  (* error                                     *)
H5TINTEGER         =  0;    (* integer types                             *)
H5TFLOAT           =  1;    (* floating-point types                      *)
H5TTIME            =  2;    (* date and time types                       *)
H5TSTRING          =  3;    (* character string types                    *)
H5TBITFIELD        =  4;    (* bit field types                           *)
H5TOPAQUE          =  5;    (* opaque types                              *)
H5TCOMPOUND        =  6;    (* compound types                            *)
H5TREFERENCE       =  7;    (* reference types                           *)
H5TENUM            =  8;    (* enumeration types                         *)
H5TVLEN            =  9;    (* Variable-Length types                     *)
H5TARRAY           = 10;    (* Array types                               *)

H5TNCLASSES        = 11;    (* the number of defined data type classes   *)
 
H5TDIRDEFAULT     = 0;    (* default direction is inscendent            *)
H5TDIRASCEND      = 1;    (* in inscendent order                        *)
H5TDIRDESCEND     = 2;    (* in descendent order                        *)

H5TCSETASCII    = 0;   		(* US ASCII                           *)
H5TCSETUTF8     = 1;   		(* UTF-8 Unicode encoding		      *)
H5TVARIABLE 	= (-1);
H5TSTRNULLTERM  = 0;   		(* null terminate like in C           *)
H5TSTRNULLPAD   = 1;   		(* pad with nulls                     *)
H5TSTRSPACEPAD  = 2;   		(* pad with spaces like in Fortran    *)

h5builtintypes = {
	H5TIEEEF32BE,
	H5TIEEEF32LE,
	H5TIEEEF64BE,
	H5TIEEEF64LE,
	
	H5TSTDI8BE,
	H5TSTDI8LE,
	H5TSTDI16BE,
	H5TSTDI16LE,
	H5TSTDI32BE,
	H5TSTDI32LE,
	H5TSTDI64BE,
	H5TSTDI64LE,
	H5TSTDU8BE,
	H5TSTDU8LE,
	H5TSTDU16BE,
	H5TSTDU16LE,
	H5TSTDU32BE,
	H5TSTDU32LE,
	H5TSTDU64BE,
	H5TSTDU64LE,
	H5TSTDB8BE,
	H5TSTDB8LE,
	H5TSTDB16BE,
	H5TSTDB16LE,
	H5TSTDB32BE,
	H5TSTDB32LE,
	H5TSTDB64BE,
	H5TSTDB64LE,
	H5TSTDREFOBJ,
	H5TSTDREFDSETREG,

	H5TNATIVECHAR,
	H5TNATIVESCHAR,
	H5TNATIVEUCHAR,
	H5TNATIVESHORT,
	H5TNATIVEUSHORT,
	H5TNATIVEINT,
	H5TNATIVEUINT,
	H5TNATIVELONG,
	H5TNATIVEULONG,
	H5TNATIVELLONG,
	H5TNATIVEULLONG,
	H5TNATIVEFLOAT,
	H5TNATIVEDOUBLE,
	
	H5TNATIVEINT8,
	H5TNATIVEUINT8,
	H5TNATIVEINT16,
	H5TNATIVEUINT16,
	H5TNATIVEINT32,
	H5TNATIVEUINT32,
	H5TNATIVEINT64,
	H5TNATIVEUINT64,

	H5TCS1,
	H5TFORTRANS1,

	H5TMREAL32,
	H5TMREAL64,
	H5TMCOMPLEX64,
	H5TMCOMPLEX128,
	H5TMCOMPLEX,
	H5TMSTRING
};
		
(* Defines from $HDF5/src/H5Zpublic.h *)

(* Filter IDs *)
H5ZFILTERERROR     	= (-1);     (* no filter                     *)
H5ZFILTERNONE      	= 0;    	(* reserved indefinitely         *)
H5ZFILTERDEFLATE   	= 1;    	(* deflation like gzip           *)
H5ZFILTERSHUFFLE   	= 2;    	(* shuffle the data              *)
H5ZFILTERFLETCHER32 	= 3;    	(* fletcher32 checksum of EDC    *)
H5ZFILTERSZIP       	= 4;    	(* szip compression              *)
H5ZFILTERNBIT       	= 5;    	(* n-bit compression             *)
H5ZFILTERSCALEOFFSET 	= 6;    	(* scale+offset compression      *)
H5ZFILTERRESERVED    	= 256;    	(* filter ids below this value are reserved for library use *)
H5ZFILTERMAX         	= 65535;  	(* maximum filter id             *)

(* Special parameters for szip compression *)
H5SZIPALLOWK13OPTIONMASK = 1;
H5SZIPCHIPOPTIONMASK     = 2;
H5SZIPECOPTIONMASK       = 4;
H5SZIPNNOPTIONMASK       = 32;
H5SZIPMAXPIXELSPERBLOCK  = 32;

(* Special parameters for ScaleOffset filter *)
H5ZSOINTMINBITSDEFAULT  = 0;
H5ZSOFLOATDSCALE 		= 0; 
H5ZSOFLOATESCALE 		= 1; (* This is not currently implemented by the library. *)
H5ZSOINT          		= 2;

(**********************************************************************************************************************)
(***********************************           End of HDF5 Constants               ************************************)
(**********************************************************************************************************************)


(* ::Section:: *)
(* Private context *)

Begin["`Private`"]

$InitHDF5Tools = False;

$ThisDirectory = FileNameDrop[$InputFileName, -1]
$BaseLibraryDirectory = FileNameJoin[{$ThisDirectory, "LibraryResources", $SystemID}];
$HDF5Library := $HDF5Library = FindLibrary["HDF5Tools"];
$MessageHead = Import;
$UseComplexType = True;
$32bitQ = BitLength[Developer`$MaxMachineInteger + 1] == 32;

$MTypeInteger = 2;
$MTypeReal = 3;
$MTypeComplex = 4;

safeLibraryLoad[debug_, lib_] :=
	Quiet[
		Check[
			LibraryLoad[lib],
			If[TrueQ[debug],
				Print["Failed to load ", lib]
			];
			Throw[$InitHDF5Tools = $Failed]
		]
	]
safeLibraryFunctionLoad[debug_, args___] :=
	Quiet[
		Check[
			LibraryFunctionLoad[$HDF5Library, args],
			If[TrueQ[debug],
				Print["Failed to load the function ", First[{args}], " from ", $HDF5Library]
			];
			Throw[$InitHDF5Tools = $Failed]
		]
	]

CheckHDF5[f_, args___] := Replace[Quiet[f[args]],  errorResult: Except[_?Internal`NonNegativeIntegerQ] :> $Failed];
CheckHandle[handle_, f_, args___] := Replace[Quiet[f[args]], { id_?Internal`NonNegativeIntegerQ :> handle[id], _ :> $Failed}];
CheckTrue[f_, args___] := Replace[Quiet[f[args]], { id_?Internal`NonNegativeIntegerQ :> True, _ :> $Failed}];

(* ::Subsection:: *)
(* HDF5ToolsInit *)

HDF5ToolsInit[debug_:False] := If[TrueQ[$InitHDF5Tools],
	$InitHDF5Tools
	,
	$InitHDF5Tools = Catch[Block[{$LibraryPath = Prepend[$LibraryPath, $BaseLibraryDirectory]},
		safeLibraryLoad[debug, $HDF5Library];
		
		Off[LibraryFunction::overload]; (* we deliberately overload some functions, no need for warning *)
		
		(*****************************************************************************************************)
		(*****************************************************************************************************)
		(***************  Low-level functions (mostly direct wrappers of HDF5 C API)  ************************)
		(*****************************************************************************************************)
		(*****************************************************************************************************)
		
		(****************************************************************************
		 * The HDF5 H5A: Attribute Interface
		 * See also: http://www.hdfgroup.org/HDF5/doc/RM/RM_H5A.html
		 ****************************************************************************)
		
		h5acreate = safeLibraryFunctionLoad[debug, "LL_H5Acreate", {Integer, "UTF8String", Integer, Integer, Integer, Integer}, Integer];

		h5aexists = safeLibraryFunctionLoad[debug, "LL_H5Aexists", {Integer, "UTF8String"}, Integer];

		h5aopen = safeLibraryFunctionLoad[debug, "LL_H5Aopen", {Integer, "UTF8String"}, Integer];

		h5adelete = safeLibraryFunctionLoad[debug, "LL_H5Adelete", {Integer, "UTF8String"}, Integer];

		h5agetnamebyidx = safeLibraryFunctionLoad[debug, "LL_H5Aget_name_by_idx", {Integer, "UTF8String", Integer, Integer, Integer}, "UTF8String"];
		
		h5agetspace = safeLibraryFunctionLoad[debug, "LL_H5Aget_space", {Integer}, Integer];
		
		h5agettype = safeLibraryFunctionLoad[debug, "LL_H5Aget_type", {Integer}, Integer];
		
		h5areadnumericarray = safeLibraryFunctionLoad[debug, "LL_H5Aread_NumericArray", {Integer, Integer}, NumericArray];
		
		h5areadstringlist = safeLibraryFunctionLoad[debug, "LL_H5Aread_String", LinkObject, LinkObject];
		
		h5areadstrings[attrId_Integer, memtypeId_Integer] := readFromLinkObject[h5areadstringlist, attrId, memtypeId];
		
		h5areadcompoundlist = safeLibraryFunctionLoad[debug, "LL_H5Aread_Compound", LinkObject, LinkObject];
		
		h5areadcompounds[attrId_Integer, memtypeId_Integer] := readFromLinkObject[h5areadcompoundlist, attrId, memtypeId];
		
		h5areadbytelist = safeLibraryFunctionLoad[debug, "LL_H5Aread_RawByte", LinkObject, LinkObject];
		
		h5areadrawbytes[attrId_Integer, memtypeId_Integer] := readRawBytes[h5areadbytelist, attrId, memtypeId];
		
		h5areadarraylist = safeLibraryFunctionLoad[debug, "LL_H5Aread_Array", LinkObject, LinkObject];
		
		h5areadarrays[attrId_Integer, memtypeId_Integer] := readArrays[h5areadarraylist, attrId, memtypeId];
		
		h5areadtensor = safeLibraryFunctionLoad[debug, "LL_H5Aread_Tensor", {Integer, Integer}, {_, _}];
		
		h5areadtensorint[attrId_Integer] := h5areadtensor[attrId, $MTypeInteger];
			
		h5areadtensorreal[attrId_Integer] := h5areadtensor[attrId, $MTypeReal];
		
		h5areadtensorcomplex[attrId_Integer] := h5areadtensor[attrId, $MTypeComplex];
			
		h5awritearray = safeLibraryFunctionLoad[debug, "LL_H5Awrite_Array", LinkObject, LinkObject];
		
		h5awritecompound = safeLibraryFunctionLoad[debug, "LL_H5Awrite_Compound", LinkObject, LinkObject];
		
		h5awritecomplex = safeLibraryFunctionLoad[debug, "LL_H5Awrite_Complex", {Integer, Integer, {Complex, _, "Constant"}}, Integer];
		
		h5awriteinteger = safeLibraryFunctionLoad[debug, "LL_H5Awrite_Integer", {Integer, Integer, {Integer, _, "Constant"}}, Integer];
		
		h5awritereal = safeLibraryFunctionLoad[debug, "LL_H5Awrite_Real", {Integer, Integer, {Real, _, "Constant"}}, Integer];
		
		h5awritestring = safeLibraryFunctionLoad[debug, "LL_H5Awrite_String", LinkObject, LinkObject];
		
		h5awritenumericarray = safeLibraryFunctionLoad[debug, "LL_H5Awrite_NumericArray", {Integer, Integer, {NumericArray, "Constant"}}, Integer];
		
		h5awriterawbyte = safeLibraryFunctionLoad[debug, "LL_H5Awrite_RawByte", LinkObject, LinkObject];
		
		h5aclose = safeLibraryFunctionLoad[debug, "LL_H5Aclose", {Integer}, Integer];
	
		(****************************************************************************
		 * The HDF5 H5D: Datasets Interface
		 * See: http://www.hdfgroup.org/HDF5/doc/RM/RM_H5D.html
		 ****************************************************************************)
		
		h5dcreate = safeLibraryFunctionLoad[debug, "LL_H5Dcreate", {Integer, "UTF8String", Integer, Integer, Integer, Integer, Integer}, Integer];
		
		h5dopen = safeLibraryFunctionLoad[debug, "LL_H5Dopen", {Integer, "UTF8String", Integer}, Integer];
		
		h5dgetspace = safeLibraryFunctionLoad[debug, "LL_H5Dget_space", {Integer}, Integer];
		
		h5dgetstoragesizeraw = safeLibraryFunctionLoad[debug, "LL_H5Dget_storage_size", {Integer}, NumericArray];
		
		h5dgettype = safeLibraryFunctionLoad[debug, "LL_H5Dget_type", {Integer}, Integer];
		
		h5dgetcreateplist = safeLibraryFunctionLoad[debug, "LL_H5Dget_create_plist", {Integer}, Integer];
		
		h5dreadstringlist = safeLibraryFunctionLoad[debug, "LL_H5Dread_String", LinkObject, LinkObject];
		
		h5dreadstrings[dsetId_Integer, memtypeId_Integer, memspaceId_Integer, dspaceId_Integer, xferPlistId_Integer] := 
			readFromLinkObject[h5dreadstringlist, dsetId, memtypeId, memspaceId, dspaceId, xferPlistId]; 
		 
		h5dreadarraylist = safeLibraryFunctionLoad[debug, "LL_H5Dread_Array", LinkObject, LinkObject];
		
		h5dreadarrays[dsetId_Integer, memtypeId_Integer, memspaceId_Integer, dspaceId_Integer, xferPlistId_Integer] := 
			readArrays[h5dreadarraylist, dsetId, memtypeId, memspaceId, dspaceId, xferPlistId]; 
		 
		h5dreadcompoundlist = safeLibraryFunctionLoad[debug, "LL_H5Dread_Compound", LinkObject, LinkObject];
		
		h5dreadcompounds[dsetId_Integer, memtypeId_Integer, memspaceId_Integer, dspaceId_Integer, xferPlistId_Integer] := 
			readFromLinkObject[h5dreadcompoundlist, dsetId, memtypeId, memspaceId, dspaceId, xferPlistId]; 
		
		h5dreadbytelist = safeLibraryFunctionLoad[debug, "LL_H5Dread_RawByte", LinkObject, LinkObject];
		
		h5dreadrawbytes[dsetId_Integer, memtypeId_Integer, memspaceId_Integer, dspaceId_Integer, xferPlistId_Integer] := 
			readRawBytes[h5dreadbytelist, dsetId, memtypeId, memspaceId, dspaceId, xferPlistId];
		
		h5dreadnumericarray = safeLibraryFunctionLoad[debug, "LL_H5Dread_NumericArray", {Integer, Integer, Integer, Integer, Integer}, {NumericArray}];
		
		h5dreadnumericarrayinto = safeLibraryFunctionLoad[debug, "LL_H5Dread_NumericArray", {Integer, Integer, Integer, Integer, Integer, {NumericArray, "Constant"}}, Integer];
		
		h5dreadtensor = safeLibraryFunctionLoad[debug, "LL_H5Dread_Tensor", {Integer, Integer, Integer, Integer, Integer}, {_, _}];
		
		h5dreadtensorint[dsetId_Integer, memspaceId_Integer, dspaceId_Integer, xferPlistId_Integer] := 
			h5dreadtensor[dsetId, memspaceId, dspaceId, xferPlistId, $MTypeInteger];
			
		h5dreadtensorreal[dsetId_Integer, memspaceId_Integer, dspaceId_Integer, xferPlistId_Integer] := 
			h5dreadtensor[dsetId, memspaceId, dspaceId, xferPlistId, $MTypeReal];
		
		h5dreadtensorcomplex[dsetId_Integer, memspaceId_Integer, dspaceId_Integer, xferPlistId_Integer] := 
			h5dreadtensor[dsetId, memspaceId, dspaceId, xferPlistId, $MTypeComplex];
			
		h5dsetextent = safeLibraryFunctionLoad[debug, "LL_H5Dset_extent", {Integer, {Integer, 1, "Constant"}}, Integer];
		
		h5dwriteinteger = safeLibraryFunctionLoad[debug, "LL_H5Dwrite_Integer", {Integer, Integer, Integer, Integer, Integer, {Integer, _, "Constant"}}, Integer];
		
		h5dwritereal = safeLibraryFunctionLoad[debug, "LL_H5Dwrite_Real", {Integer, Integer, Integer, Integer, Integer, {Real, _, "Constant"}}, Integer];
		
		h5dwritecomplex = safeLibraryFunctionLoad[debug, "LL_H5Dwrite_Complex", {Integer, Integer, Integer, Integer, Integer, {Complex, _, "Constant"}}, Integer];
		
		h5dwritestring = safeLibraryFunctionLoad[debug, "LL_H5Dwrite_String", LinkObject, LinkObject];
		
		h5dwritearray = safeLibraryFunctionLoad[debug, "LL_H5Dwrite_Array", LinkObject, LinkObject];
		
		h5dwritecompound = safeLibraryFunctionLoad[debug, "LL_H5Dwrite_Compound", LinkObject, LinkObject];
		
		h5dwritenumericarray = safeLibraryFunctionLoad[debug, "LL_H5Dwrite_NumericArray", {Integer, Integer, Integer, Integer, Integer, {NumericArray, "Constant"}}, Integer];
		
		h5dwriterawbyte = safeLibraryFunctionLoad[debug, "LL_H5Dwrite_RawByte", LinkObject, LinkObject];
		
		h5dclose = safeLibraryFunctionLoad[debug, "LL_H5Dclose", {Integer}, Integer];
	
		(****************************************************************************
		 * The HDF5 H5E: Error Interface
		 * See: http://www.hdfgroup.org/HDF5/doc/RM/RM_H5E.html
		 ****************************************************************************)
		
		h5esetfile = safeLibraryFunctionLoad[debug, "LL_H5Eset_file", {"UTF8String"}, Integer];
		
		h5egeterrmsg = safeLibraryFunctionLoad[debug, "LL_H5Eget_err_msg", {}, "UTF8String"];
		
		(****************************************************************************
		 * The HDF5 H5F: File Interface
		 * See: http://www.hdfgroup.org/HDF5/doc/RM/RM_H5F.html
		 ****************************************************************************)
		
		h5fishdf5 = safeLibraryFunctionLoad[debug, "LL_H5Fis_hdf5", {"UTF8String"}, "Boolean"];
		
		h5fcreate = safeLibraryFunctionLoad[debug, "LL_H5Fcreate", {"UTF8String", Integer}, Integer];
		
		h5fflush = safeLibraryFunctionLoad[debug, "LL_H5Fflush", {Integer, Integer}, Integer];
		
		h5fgetfilesizeraw = safeLibraryFunctionLoad[debug, "LL_H5Fget_filesize", {Integer}, NumericArray];
		
		h5fopen = safeLibraryFunctionLoad[debug, "LL_H5Fopen", {"UTF8String", Integer}, Integer];
		
		h5fclose = safeLibraryFunctionLoad[debug, "LL_H5Fclose", {Integer}, Integer];
	
		(****************************************************************************
		 * The HDF5 H5G: Group Interface
		 * See: http://www.hdfgroup.org/HDF5/doc/RM/RM_H5G.html
		 ****************************************************************************)
		
		h5gcreate = safeLibraryFunctionLoad[debug, "LL_H5Gcreate", {Integer, "UTF8String", Integer, Integer, Integer}, Integer];
		
		h5gopen = safeLibraryFunctionLoad[debug, "LL_H5Gopen", {Integer, "UTF8String"}, Integer];
		
		h5ggetinfonlinksraw = safeLibraryFunctionLoad[debug, "LL_H5Gget_info_nlinks", {Integer}, NumericArray];
		
		h5gclose = safeLibraryFunctionLoad[debug, "LL_H5Gclose", {Integer}, Integer];
	
		(****************************************************************************
		 * The HDF5 H5L: Link Interface
		 * See: http://www.hdfgroup.org/HDF5/doc/RM/RM_H5L.html
		 ****************************************************************************)
		
		h5lcreatehard = safeLibraryFunctionLoad[debug, "LL_H5Lcreate_hard", {Integer, "UTF8String", Integer, "UTF8String", Integer, Integer}, Integer];
		
		h5lcreatesoft = safeLibraryFunctionLoad[debug, "LL_H5Lcreate_soft", {"UTF8String", Integer, "UTF8String", Integer, Integer}, Integer];

		h5ldelete = safeLibraryFunctionLoad[debug, "LL_H5Ldelete", {Integer, "UTF8String", Integer}, Integer];

		h5lexists = safeLibraryFunctionLoad[debug, "LL_H5Lexists", {Integer, "UTF8String", Integer}, Integer];
		
		h5lgetinfotype = safeLibraryFunctionLoad[debug, "LL_H5Lget_info_type", {Integer, "UTF8String", Integer}, Integer];
		
		h5lgetnamebyidx = safeLibraryFunctionLoad[debug, "LL_H5Lget_name_by_idx", {Integer, "UTF8String", Integer, Integer, Integer}, "UTF8String"];
	
		(****************************************************************************
		 * The HDF5 H5LT: "Lite" Interface
		 * See: See: https://support.hdfgroup.org/HDF5/doc/HL/RM_H5LT.html
		 ****************************************************************************)
		
		h5ltpathvalid = safeLibraryFunctionLoad[debug, "LL_H5LTpath_valid", {Integer, "UTF8String", True|False}, True|False];
		
		h5lttexttodtype = safeLibraryFunctionLoad[debug, "LL_H5LTtext_to_dtype", {"UTF8String"}, Integer];
		
		h5ltdtypetotext = safeLibraryFunctionLoad[debug, "LL_H5LTdtype_to_text", {Integer}, "UTF8String"];
		
		(****************************************************************************
		 * The HDF5 H5O: Object Interface
		 * See: http://www.hdfgroup.org/HDF5/doc/RM/RM_H5O.html
		 ****************************************************************************)
		
		h5oexistsbyname = safeLibraryFunctionLoad[debug, "LL_H5Oexists_by_name", {Integer, "UTF8String", Integer}, Integer];
		
		h5oopen = safeLibraryFunctionLoad[debug, "LL_H5Oopen", {Integer, "UTF8String", Integer}, Integer];
		
		h5ogetinfoaddrraw = safeLibraryFunctionLoad[debug, "LL_H5Oget_info_addr", {Integer}, NumericArray];
		
		h5ogetinfonumattrsraw = safeLibraryFunctionLoad[debug, "LL_H5Oget_info_num_attrs", {Integer}, NumericArray];
		
		h5ogetinfotype = safeLibraryFunctionLoad[debug, "LL_H5Oget_info_type", {Integer}, Integer];
		
		h5ogetinfobynameaddrraw = safeLibraryFunctionLoad[debug, "LL_H5Oget_info_by_name_addr", {Integer, "UTF8String"}, NumericArray];
		
		h5oclose = safeLibraryFunctionLoad[debug, "LL_H5Oclose", {Integer}, Integer];
	
		(****************************************************************************
		 * The HDF5 H5P: Property List Interface
		 * http://www.hdfgroup.org/HDF5/doc/RM/RM_H5P.html
		 ****************************************************************************)
		
		h5pgetclassid = safeLibraryFunctionLoad[debug, "LL_H5Pget_class_id", {Integer}, Integer];
		
		Evaluate[h5pbuiltinclasses] = h5pgetclassid /@ Range[0, 16];
		
		h5pcreate = safeLibraryFunctionLoad[debug, "LL_H5Pcreate", {Integer}, Integer];
		
		h5pgetlayout = safeLibraryFunctionLoad[debug, "LL_H5Pget_layout", {Integer}, Integer];
		
		h5pgetchunk = safeLibraryFunctionLoad[debug, "LL_H5Pget_chunk", {Integer, Integer}, {Integer, 1}];
		
		h5pgetfilterid = safeLibraryFunctionLoad[debug, "LL_H5Pget_filter_id", {Integer, Integer}, Integer];
		
		h5pgetnfilters = safeLibraryFunctionLoad[debug, "LL_H5Pget_nfilters", {Integer}, Integer];
		
		h5psetcharencoding = safeLibraryFunctionLoad[debug, "LL_H5Pset_char_encoding", {Integer, Integer}, Integer];

		h5psetlayout = safeLibraryFunctionLoad[debug, "LL_H5Pset_layout", {Integer, Integer}, Integer];

		h5psetchunk = safeLibraryFunctionLoad[debug, "LL_H5Pset_chunk", {Integer, Integer, {Integer, 1, "Constant"}}, Integer];
		
		h5psetcreateintermediategroup = safeLibraryFunctionLoad[debug, "LL_H5Pset_create_intermediate_group", {Integer, Integer}, Integer];
		
		h5psetdeflate = safeLibraryFunctionLoad[debug, "LL_H5Pset_deflate", {Integer, Integer}, Integer];
		
		h5psetfletcher32 = safeLibraryFunctionLoad[debug, "LL_H5Pset_fletcher32", {Integer}, Integer];
		
		h5psetscaleoffset = safeLibraryFunctionLoad[debug, "LL_H5Pset_scaleoffset", {Integer, Integer, Integer}, Integer]; 
		
		h5psetshuffle = safeLibraryFunctionLoad[debug, "LL_H5Pset_shuffle", {Integer}, Integer];
		
		h5psetszip = safeLibraryFunctionLoad[debug, "LL_H5Pset_szip", {Integer, Integer, Integer}, Integer];
		
		h5pclose = safeLibraryFunctionLoad[debug, "LL_H5Pclose", {Integer}, Integer];
		
		(****************************************************************************
		 * The HDF5 H5S: Dataspace Interface
		 * See: http://www.hdfgroup.org/HDF5/doc/RM/RM_H5S.html
		 ****************************************************************************) 
		
		h5screate = safeLibraryFunctionLoad[debug, "LL_H5Screate", {Integer}, Integer];
		
		h5scopy = safeLibraryFunctionLoad[debug, "LL_H5Scopy", {Integer}, Integer];
		
		h5screatesimple = safeLibraryFunctionLoad[debug, "LL_H5Screate_simple", {Integer, {Integer, 1, "Constant"}, {Integer, 1, "Constant"}}, Integer];
		
		h5screatesimplen[rank_Integer, dims_List] := h5screatesimple[rank, dims, dims];
		
		h5sgetsimpleextentdims = safeLibraryFunctionLoad[debug, "LL_H5Sget_simple_extent_dims", {Integer}, {Integer, 1}];
		
		h5sgetsimpleextentmaxdims = safeLibraryFunctionLoad[debug, "LL_H5Sget_simple_extent_max_dims", {Integer}, {Integer, 1}];
		
		h5sgetsimpleextentndims = safeLibraryFunctionLoad[debug, "LL_H5Sget_simple_extent_ndims", {Integer}, Integer];
		
		h5sgetsimpleextentnpointsraw = safeLibraryFunctionLoad[debug, "LL_H5Sget_simple_extent_npoints", {Integer}, NumericArray];
		
		h5sgetsimpleextenttype = safeLibraryFunctionLoad[debug, "LL_H5Sget_simple_extent_type", {Integer}, Integer];
		
		h5sselecthyperslab = safeLibraryFunctionLoad[debug, "LL_H5Sselect_hyperslab", 
		 	 {Integer, Integer, {Integer, 1, "Constant"}, {Integer, 1, "Constant"}, {Integer, 1, "Constant"}, {Integer, 1, "Constant"}}, Integer];
		
		h5sselectelements = safeLibraryFunctionLoad[debug, "LL_H5Sselect_elements", {Integer, Integer, Integer, {NumericArray, "Constant"}}, Integer];
		
		h5sclose = safeLibraryFunctionLoad[debug, "LL_H5Sclose", {Integer}, Integer];
	
		(****************************************************************************
		 * The HDF5 H5T: Datatype Interface
		 * See: http://www.hdfgroup.org/HDF5/doc/RM/RM_H5T.html
		 ****************************************************************************)
		
		h5tgettype = safeLibraryFunctionLoad[debug, "LL_H5Tget_type", {Integer}, Integer];
		
		If[ListQ[h5builtintypenames], Clear /@ h5builtintypenames];	
		h5builtintypenames = SymbolName /@ h5builtintypes;
		
		Evaluate[h5builtintypes] = h5tgettype /@ Range[0, Length[h5builtintypenames] - 1];
		
		h5tregistercomplex = safeLibraryFunctionLoad[debug, "LL_H5Tregister_complex", {"Boolean", "Boolean", "UTF8String", "UTF8String"}, {Integer, 1}];
		
		h5tconvert = safeLibraryFunctionLoad[debug, "LL_H5Tconvert", {Integer, Integer, NumericArray, Integer}, NumericArray];
		
		h5tcreate = safeLibraryFunctionLoad[debug, "LL_H5Tcreate", {Integer, Integer}, Integer];
		
		h5tenumcreate = safeLibraryFunctionLoad[debug, "LL_H5Tenum_create", {Integer}, Integer];
		
		h5tcreatestr = safeLibraryFunctionLoad[debug, "LL_H5Tcreate_string_type", {Integer, Integer, Integer, Integer}, Integer];
		
		h5tarraycreate = safeLibraryFunctionLoad[debug, "LL_H5Tarray_create", {Integer, Integer, {Integer, 1, "Constant"}}, Integer];
		
		h5tequal = safeLibraryFunctionLoad[debug, "LL_H5Tequal", {Integer, Integer}, True|False];
		
		h5tinsert = safeLibraryFunctionLoad[debug, "LL_H5Tinsert", {Integer, "UTF8String", Integer, Integer}, Integer];
		
		h5tenuminsert = safeLibraryFunctionLoad[debug, "LL_H5Tenum_insert", {Integer, "UTF8String", Integer}, Integer];
		
		h5tcommit = safeLibraryFunctionLoad[debug, "LL_H5Tcommit", {Integer, "UTF8String", Integer, Integer, Integer, Integer}, Integer];
		
		h5tcopy = safeLibraryFunctionLoad[debug, "LL_H5Tcopy", {Integer}, Integer];
		
		h5topen = safeLibraryFunctionLoad[debug, "LL_H5Topen", {Integer, "UTF8String", Integer}, Integer];
		
		h5tgetclass = safeLibraryFunctionLoad[debug, "LL_H5Tget_class", {Integer}, Integer];
		
		h5tgetnativetype = safeLibraryFunctionLoad[debug, "LL_H5Tget_native_type", {Integer, Integer}, Integer];
		
		h5tgetarrayndims = safeLibraryFunctionLoad[debug, "LL_H5Tget_array_ndims", {Integer}, Integer];
		
		h5tgetarraydims = safeLibraryFunctionLoad[debug, "LL_H5Tget_array_dims", {Integer}, {Integer, 1}];
		
		h5tgetnmembers = safeLibraryFunctionLoad[debug, "LL_H5Tget_nmembers", {Integer}, Integer];
		
		h5tgetmembername = safeLibraryFunctionLoad[debug, "LL_H5Tget_member_name", {Integer, Integer}, "UTF8String"];
		
		h5tgetmemberoffset = safeLibraryFunctionLoad[debug, "LL_H5Tget_member_offset", {Integer, Integer}, Integer];
		
		h5tgetmembertype = safeLibraryFunctionLoad[debug, "LL_H5Tget_member_type", {Integer, Integer}, Integer];
		
		h5tgetmembervalueraw = safeLibraryFunctionLoad[debug, "LL_H5Tget_member_value", {Integer, Integer}, NumericArray];
		
		h5tgetnamestring = safeLibraryFunctionLoad[debug, "LL_H5Tget_nameString", {Integer}, "UTF8String"];
		
		h5tgettag = safeLibraryFunctionLoad[debug, "LL_H5Tget_tag", {Integer}, "UTF8String"];
		
		h5tsettag = safeLibraryFunctionLoad[debug, "LL_H5Tset_tag", {Integer, "UTF8String"}, Integer];
		
		h5tgetsize = safeLibraryFunctionLoad[debug, "LL_H5Tget_size", {Integer}, Integer];
		
		h5tgetsuper = safeLibraryFunctionLoad[debug, "LL_H5Tget_super", {Integer}, Integer];
		
		h5tclose = safeLibraryFunctionLoad[debug, "LL_H5Tclose", {Integer}, Integer]; 	
		 	
		(****************************************************************************
		 * The HDF5 H5Z: Filter and Compression Interface
		 * See: https://support.hdfgroup.org/HDF5/doc/RM/RM_H5Z.html
		 ****************************************************************************)
		
		h5zfilteravail = safeLibraryFunctionLoad[debug, "LL_H5Zfilter_avail", {Integer}, True|False];
		
		h5zgetfilterinfo = safeLibraryFunctionLoad[debug, "LL_H5Zget_filter_info", {Integer}, {Integer, 1}];
		 
		h5traverse = safeLibraryFunctionLoad[debug, "LL_H5traverse", LinkObject, LinkObject];

		h5traversetolevel = safeLibraryFunctionLoad[debug, "LL_H5traverse_to_level", LinkObject, LinkObject];
		
		On[LibraryFunction::overload]; (* restore message *)
		(*****************************************************************************************************)
		(*****************************************************************************************************)
		(****  "Medium-level" functions (reading/writing datasets and attributes + some helper functions) ****)
		(*****************************************************************************************************)
		(*****************************************************************************************************)
		
		readFromLinkObject[readingF_, args__]:= 
			Module[{tmp, dims, objs},
				tmp = readingF[args];
				If[FailedQ[tmp] || !MatchQ[tmp, {_List, _List}],
					Return[$Failed]
					,
					{dims, objs} = tmp;
					If[objs === {}, Return[{}]];
					Return @ ArrayReshape[objs, dims]
				]
			];
		
		readArrays[readingF_, args__]:= 
			Module[{tmp, dims, arrays, arrDims},
				If[FailedQ[tmp = readingF[args]] || !MatchQ[tmp, {_List, _List}],
					Return[$Failed]
					,
					{dims, arrays} = tmp;
					If[arrays === {}, Return[{}]];
					arrDims = Dimensions[arrays][[2;;]];
					Return @ ArrayReshape[arrays, dims~Join~arrDims]
				]
			];
			
		readRawBytes[readingF_, args__]:= 	
			Module[{dims, bytes, tmp},
				If[FailedQ[tmp = readingF[args]] || !MatchQ[tmp, {_List, _List}],
					Return[$Failed]
					,
					{dims, bytes} = tmp;
					If[bytes === {}, Return[{}]];
					Return @ If[Length[dims] < 2,
						{ ByteArray[bytes] }
						,
						Map[ByteArray, ArrayReshape[bytes, dims], {-2}]
					]
				]
			];
			
		h5areadint[attrId_Integer, memtypeId_Integer] := 
			Module[{bType, res, raType},
				bType = If[h5tgetclass[memtypeId] == H5TENUM, h5tgetsuper[memtypeId], memtypeId];					
				If[FailureQ[raType = memTypeToNumericArrayType[bType]], Return[$Failed]];
				res = h5areadnumericarray[attrId, memtypeId];
				If[h5tgetclass[memtypeId] == H5TENUM, h5tclose[bType]];
				res
			];
			
		h5aread[attrId_Integer, memtypeId_Integer] := 
			Module[{typeClass = h5tgetclass[memtypeId], dspace, res},
				If[h5tiscomplex[memtypeId],
					res = h5areadnumericarray[attrId, memtypeId]
					,
					Switch[typeClass,
						H5TSTRING, 
							res = h5areadstrings[attrId, memtypeId],
						H5TOPAQUE, 
							res = h5areadrawbytes[attrId, memtypeId],
						H5TCOMPOUND,
							res = h5areadcompounds[attrId, memtypeId],
						H5TARRAY,
							res = h5areadarrays[attrId, memtypeId],
						H5TFLOAT,
							res = h5areadnumericarray[attrId, memtypeId],
						H5TINTEGER | H5TENUM,
							res = h5areadint[attrId, memtypeId],
						_,
							Message[Import::general, ErrUnsuppClass[memtypeId]]; 
							Return[$Failed]
					]
				];
				dspace = h5agetspace[attrId];
				If[Not[FailedQ[res]] && h5sgetsimpleextenttype[dspace] == H5SSCALAR,
					res = First @ NormalizeNumericArray[res];
				];
				h5sclose[dspace];
				Return[res];
			];
			
		h5agetwritespacerank[attrId_Integer] := Module[{s, d},
			s = h5agetspace[attrId];
			d = h5sgetsimpleextentndims[s];
			h5sclose[s];
			Return[d];
		];
		
		h5awrite[attrId_Integer, memtypeId_Integer, inData_?ArrayQ] := 
			Module[{data, packedData, isMComplex, typeClass, flatData, lengths, r},
				typeClass = h5tgetclass[memtypeId];
				isMComplex = h5tiscomplex[memtypeId];
				data = Replace[inData, p_ByteArray :> Normal[p], All];
				Which[
					typeClass == H5TSTRING,
						Return @ h5awritestring[attrId, memtypeId, Flatten[data]],
					typeClass == H5TOPAQUE,
						Return @ h5awriterawbyte[attrId, memtypeId, Flatten[data]],
					typeClass == H5TARRAY,
						r = h5agetwritespacerank[attrId];
						Return @ h5awritearray[attrId, memtypeId, Flatten[data, Max[0, r-1]]],
					typeClass == H5TCOMPOUND && Not[isMComplex],
						If[!ArrayQ[data, _, AssociationQ],
							Message[Export::general, ErrCmpdAssArr[data]];
							Return[$Failed]
						];
						flatData = Flatten[N[data]];
						lengths = Length /@ flatData;
						If[Min[lengths] != Max[lengths],
							Message[Export::general, ErrCmpEqLength[data]];
							Return[$Failed]
						];
						Return @ h5awritecompound[attrId, memtypeId, flatData]
				];
				packedData = Developer`ToPackedArray[data, h5tToPackedArrayType[memtypeId]];
				If[!Developer`PackedArrayQ[packedData],
					Message[Export::general, ErrUnsuppWriteType];
					Return[$Failed]
				];
				Which[
					MatchQ[typeClass, H5TINTEGER | H5TENUM],
						h5awriteinteger[attrId, memtypeId, packedData],
					typeClass == H5TFLOAT,
						h5awritereal[attrId, memtypeId, packedData],
					isMComplex,
						h5awritecomplex[attrId, memtypeId, packedData],
					True,
						Message[Export::general, ErrUnsuppWriteType];
						Return[$Failed];
				]
			];
	
		h5awrite[attrId_Integer, memtypeId_Integer, data_?NumericArrayQ] := 
			Module[{raType},
				raType = NumericArrayType[data];
				If[raType =!= memTypeToNumericArrayType[memtypeId],
					Message[Export::general, ErrUnsuppWriteType];
					Return[$Failed]
				];
				h5awritenumericarray[attrId, memtypeId, data]
			];
			
		h5awrite[attrId_Integer, memtypeId_Integer, data_?AtomQ] := 
			Which[
				StringQ[data],
					h5awritestring[attrId, memtypeId, {data}],
				IntegerQ[data],
					h5awriteinteger[attrId, memtypeId, {data}],
				Internal`RealValuedNumericQ[data],
					h5awritereal[attrId, memtypeId, {data}],
				NumberQ[data],
					h5awritecomplex[attrId, memtypeId, {data}],
				True,
					Message[Export::general, ErrUnsuppWriteType];
					$Failed
			];
			
		h5awrite[attrId_Integer, memtypeId_Integer, data_] := (Message[Export::rect]; $Failed);
		
		h5dreadint[dsetId_Integer, memtypeId_Integer, memspaceId_Integer, dspaceID_Integer, xferPlistId_Integer] := 
			Module[{bType, res, raType},
				bType = If[h5tgetclass[memtypeId] == H5TENUM, h5tgetsuper[memtypeId], memtypeId];
				If[FailureQ[raType = memTypeToNumericArrayType[bType]], Return[$Failed]];
				res = h5dreadnumericarray[dsetId, memtypeId, memspaceId, dspaceID, xferPlistId];
				If[h5tgetclass[memtypeId] == H5TENUM, h5tclose[bType]];
				res
			];
			
		h5dread[dsetId_Integer, memtypeId_Integer, memspaceId_Integer, dspaceID_Integer, xferPlistId_Integer] := 
			Module[{typeClass = h5tgetclass[memtypeId], dspace, res},
				If[h5tiscomplex[memtypeId],
					res = h5dreadnumericarray[dsetId, memtypeId, memspaceId, dspaceID, xferPlistId]
					,
					Switch[typeClass,
						H5TSTRING, 
							res = h5dreadstrings[dsetId, memtypeId, memspaceId, dspaceID, xferPlistId],
						H5TOPAQUE, 
							res = h5dreadrawbytes[dsetId, memtypeId, memspaceId, dspaceID, xferPlistId],
						H5TCOMPOUND,
							res = h5dreadcompounds[dsetId, memtypeId, memspaceId, dspaceID, xferPlistId],
						H5TARRAY,
							res = h5dreadarrays[dsetId, memtypeId, memspaceId, dspaceID, xferPlistId],
						H5TFLOAT,
							res = h5dreadnumericarray[dsetId, memtypeId, memspaceId, dspaceID, xferPlistId],
						H5TENUM | H5TINTEGER,
							res = h5dreadint[dsetId, memtypeId, memspaceId, dspaceID, xferPlistId],
						_,
							Message[Import::general, ErrUnsuppClass[memtypeId]]; 
							Return[$Failed]
					]
				];
				dspace = h5dgetspace[dsetId];
				If[Not[FailedQ[res]] && (h5sgetsimpleextenttype[dspace] == H5SSCALAR),
					res = First @ NormalizeNumericArray[res];
				];
				h5sclose[dspace];
				Return[res];
			];
			
		h5dgetwritespacerank[dsetId_Integer, memspaceId_Integer, dspaceId_Integer] := 
			Module[{s, d},
				Which[
					memspaceId != H5SALL,
						Return @ h5sgetsimpleextentndims[memspaceId],
					dspaceId != H5SALL,
						Return @ h5sgetsimpleextentndims[dspaceId],
					True,
						s = h5dgetspace[dsetId];
						d = h5sgetsimpleextentndims[s];
						h5sclose[s];
						Return[d];
				];
			];
		
		h5dwrite[dsetId_Integer, memtypeId_Integer, memspaceId_Integer, dspaceId_Integer, xferPlistId_Integer, inData_?ArrayQ] := 
			Module[{packedData, flatData, lengths, typeClass, isMComplex, r, data, raType, raData},
				typeClass = h5tgetclass[memtypeId];
				isMComplex = h5tiscomplex[memtypeId];
				data = inData;
				Which[
					typeClass == H5TSTRING,
						Return @ h5dwritestring[dsetId, memtypeId, memspaceId, dspaceId, xferPlistId, Flatten[data]],
					typeClass == H5TOPAQUE,
						data = Replace[inData, p_ByteArray :> Normal[p], All];
						Return @ h5dwriterawbyte[dsetId, memtypeId, memspaceId, dspaceId, xferPlistId, Flatten[data]],
					typeClass == H5TARRAY,
						data = Replace[inData, p_ByteArray :> Normal[p], All];
						r = h5dgetwritespacerank[dsetId, memspaceId, dspaceId];
						Return @ h5dwritearray[dsetId, memtypeId, memspaceId, dspaceId, xferPlistId, Flatten[data, Max[0, r-1]]],
					typeClass == H5TCOMPOUND && Not[isMComplex],
						data = Replace[inData, p_ByteArray :> Normal[p], All];
						If[!ArrayQ[data, _, AssociationQ],
							Message[Export::general, ErrCmpdAssArr[data]];
							Return[$Failed]
						];
						flatData = Flatten @ N[data];
						lengths = Length /@ flatData;
						If[Min[lengths] != Max[lengths],
							Message[Export::general, ErrCmpEqLength[data]];
							Return[$Failed]
						];
						Return @ h5dwritecompound[dsetId, memtypeId, memspaceId, dspaceId, xferPlistId, flatData]
				];
				(* Try converting to NumericArray, this is the fastest *)
				If[Not @ FailureQ[raType = memTypeToNumericArrayType[memtypeId]],
					raData = NumericArray[data, raType];
					If[NumericArrayQ[raData], 
						Return @ h5dwritenumericarray[dsetId, memtypeId, memspaceId, dspaceId, xferPlistId, raData]
					]
				];
				(* If NumericArray failed, try at least packing *)
				packedData = Developer`ToPackedArray[data, h5tToPackedArrayType[memtypeId]];
				If[!Developer`PackedArrayQ[packedData],
					Message[Export::general, ErrUnsuppWriteType];
					Return[$Failed]
				];
				Which[
					MatchQ[typeClass, H5TINTEGER | H5TENUM],
						h5dwriteinteger[dsetId, memtypeId, memspaceId, dspaceId, xferPlistId, packedData],
					typeClass == H5TFLOAT, 
						h5dwritereal[dsetId, memtypeId, memspaceId, dspaceId, xferPlistId, packedData],
					isMComplex, 
						h5dwritecomplex[dsetId, memtypeId, memspaceId, dspaceId, xferPlistId, packedData],
					True,
						Message[Export::general, ErrUnsuppWriteType];
						Return[$Failed];
				]
			];
		
		h5dwrite[dsetId_Integer, memtypeId_Integer, memspaceId_Integer, dspaceId_Integer, xferPlistId_Integer, data_?NumericArrayQ] := 
			Module[{raType},
				raType = NumericArrayType[data];
				If[raType =!= memTypeToNumericArrayType[memtypeId],
					Message[Export::general, ErrUnsuppWriteType];
					Return[$Failed]
				];
				h5dwritenumericarray[dsetId, memtypeId, memspaceId, dspaceId, xferPlistId, data]
			];
			
		h5dwrite[dsetId_Integer, memtypeId_Integer, memspaceId_Integer, dspaceId_Integer, xferPlistId_Integer, data_] := 
			Block[{},
				Message[Export::rect];
				Return[$Failed]
			];
		
		h5dgetstoragesize[dsetId_Integer] := extractIntegerFromNumericArray[h5dgetstoragesizeraw, dsetId];
		h5fgetfilesize[fileId_Integer] := extractIntegerFromNumericArray[h5fgetfilesizeraw, fileId];
		h5ggetinfonlinks[groupId_Integer] := extractIntegerFromNumericArray[h5ggetinfonlinksraw, groupId];
		h5ogetinfoaddr[objId_Integer] := extractIntegerFromNumericArray[h5ogetinfoaddrraw, objId];
		h5ogetinfonumattrs[objId_Integer] := extractIntegerFromNumericArray[h5ogetinfonumattrsraw, objId];
		h5ogetinfobynameaddr[objId_Integer, name_String] := extractIntegerFromNumericArray[h5ogetinfobynameaddrraw, objId, name];
		h5sgetsimpleextentnpoints[spaceId_Integer] := extractIntegerFromNumericArray[h5sgetsimpleextentnpointsraw, spaceId];
		h5tgetmembervalue[type_Integer, memberIndx_Integer] := extractIntegerFromNumericArray[h5tgetmembervalueraw, type, memberIndx];
		
		h5tiscomplex64[type_Integer] := $UseComplexType && h5tequal[type, H5TMCOMPLEX64];
		h5tiscomplex128[type_Integer] := $UseComplexType && h5tequal[type, H5TMCOMPLEX128];
		h5tiscomplex[type_Integer] := $UseComplexType && (h5tequal[type, H5TMCOMPLEX64] || h5tequal[type, H5TMCOMPLEX128]);
				
		h5tToPackedArrayType[memtypeId_Integer] :=
			Block[{typeClass = h5tgetclass[memtypeId]},
				Which[
					typeClass == H5TINTEGER, Integer,
					typeClass == H5TFLOAT, Real,
					h5tiscomplex[memtypeId], Complex,
					True, Integer
				]
			];

		isOneOf[typeID_Integer, types:{_Integer..}] := AnyTrue[types, h5tequal[typeID, #]&];

		memTypeToNumericArrayType[memTypeId_Integer] :=
			Module[{baseType, res},
				Switch[h5tgetclass[memTypeId],
					H5TINTEGER,
					  	Which[
							isOneOf[memTypeId, { H5TNATIVEINT8, H5TSTDI8BE, H5TSTDI8LE }],		"Integer8",
							isOneOf[memTypeId, { H5TNATIVEUINT8, H5TSTDU8BE, H5TSTDU8LE }],	    "UnsignedInteger8",
							isOneOf[memTypeId, { H5TNATIVEINT16, H5TSTDI16BE, H5TSTDI16LE }],	"Integer16",
							isOneOf[memTypeId, { H5TNATIVEUINT16, H5TSTDU16BE, H5TSTDU16LE }],	"UnsignedInteger16",
							isOneOf[memTypeId, { H5TNATIVEINT32, H5TSTDI32BE, H5TSTDI32LE }],	"Integer32",
							isOneOf[memTypeId, { H5TNATIVEUINT32, H5TSTDU32BE, H5TSTDU32LE }],	"UnsignedInteger32",
							isOneOf[memTypeId, { H5TNATIVEINT64, H5TSTDI64BE, H5TSTDI64LE }],	"Integer64",
							isOneOf[memTypeId, { H5TNATIVEUINT64, H5TSTDU64BE, H5TSTDU64LE }],	"UnsignedInteger64",
							True, $Failed
						],
					H5TFLOAT,
						Which[
							isOneOf[memTypeId, { H5TMREAL32, H5TIEEEF32BE, H5TIEEEF32LE }],	"Real32",
							isOneOf[memTypeId, { H5TMREAL64, H5TIEEEF64BE, H5TIEEEF64LE }],	"Real64",
							True, $Failed
						],
					H5TCOMPOUND,
						Which[
							h5tiscomplex64[memTypeId],		"ComplexReal32",
							h5tiscomplex128[memTypeId],		"ComplexReal64",
							True, $Failed
						],
					 H5TENUM,
					 	baseType = h5tgetsuper[memTypeId];
					 	res = memTypeToNumericArrayType[baseType];
					 	h5tclose[baseType];
					 	res,
					 _, $Failed
				]
			];
		]; 
		True
	]
] (* HDF5ToolsInit[ ] *)
	
(* ::Subsection:: *)
(* ErrorMessages *)

ErrUnsuppClass[typeId_] := "Reading data of class " <> classToString[typeId] <> " is not currently supported.";
ErrUnsuppWriteType = "Cannot write data of unsupported datatype.";
ErrUnsuppDataType[typeId_] := "Following datatype is currently not supported: "<> ToString @ HDF5PrintType[typeId, 3];
ErrCmpEqLength[data_] := "Expecting associations of equal length in "<> ToString[data] <>".";
ErrCmpdAssArr[data_] := "Expecting a rectangular array of associations instead of "<> ToString[data] <>".";
ErrNumericArrayType[raType_] := "Expecting proper NumericArray type instead of " <> ToString[raType];
ErrFieldNames[input_] := "Field names must be Strings in " <> ToString[input];
ErrUnicodeNames[fpath_] := "File path \"" <> fpath <> "\" contains non-ASCII characters and may not be accessible on Windows. Please provide a path with ASCII characters.";
	
FailedQ[expr_] := FailureQ[expr] || (Head[expr] === LibraryFunctionError);

(* ::Subsection:: *)
(**********************************  "Public" high-level functions  **********************************)
$HDF5ToolsSetMessageHead[head_] := ($MessageHead = head)
$HDF5ToolsSetUseComplex[useQ_] := ($UseComplexType = useQ)

hasNonASCIIName[fname_String] := 
	If[StringMatchQ[$SystemID, "Windows" ~~ ___] && !PrintableASCIIQ[fname],
		If[$MessageHead === Import, 
			Message[Import::general, ErrUnicodeNames[fname]], 
			Message[Export::general, ErrUnicodeNames[fname]]
		];
		True
		,
		False
	]

extractIntegerFromNumericArray[f_, args___] :=
	Block[{val = f[args]},
		(* It's assumed that f will produce a single-element NumericArray.
		 * This is needed to exchange 64-bit integers with the paclet code *)
		If[NumericArrayQ[val],
			First[val, $Failed]
			,
			$Failed
		]
	];

(* Helper function for normalizing NumericArrays *)
NormalizeNumericArray[expr_?NumericArrayQ] := Normal[expr];
NormalizeNumericArray[expr_] := expr;

HDF5ishdf5[filename_String] :=
	Module[{},
		If[FailureQ[HDF5ToolsInit[]], Return[$Failed]];
		If[FileExistsQ[filename], 
			If[hasNonASCIIName[filename],
				$Failed
				,
				h5fishdf5[filename]
			]
			,
			Message[Evaluate[Evaluate[$MessageHead]::noopen], filename];
			$Failed
		]
	]
			
HDF5OpenFile[filename_String, mode_:H5FACCRDONLY] :=
	Block[{},
		If[FailureQ[HDF5ToolsInit[]] || hasNonASCIIName[filename], Return[$Failed]];
		Switch[mode,
			H5FACCRDONLY | H5FACCRDWR,
				If[!FileExistsQ[filename],
					Message[Evaluate[Evaluate[$MessageHead]::noopen], filename];
					Return[$Failed]
				];
				If[!h5fishdf5[filename],
					Message[Import::fmterr, "HDF5"];
					Return[$Failed]
				];
				CheckHandle[HDF5FileHandle, h5fopen, filename, mode],
			H5FACCTRUNC,
				CheckHandle[HDF5FileHandle, h5fcreate, filename, mode],
			H5FACCEXCL,
				If[FileExistsQ[filename],
					Return[$Failed]
				];
				CheckHandle[HDF5FileHandle, h5fcreate, filename, mode],
			_,
				Return[$Failed]
		]
	] (* End HDF5OpenFile[] *)
	
HDF5CloseFile[HDF5FileHandle[fileId_]] := h5fclose[fileId]

HDF5CreatePropertyList[class_Integer] := CheckHandle[HDF5PropListHandle, h5pcreate, class];

HDF5ClosePropertyList[HDF5PropListHandle[plistID_]] := h5pclose[plistID];

CreateLinkPropList[createIntermediateQ_, encoding_Integer] := 
	Block[{plistHndl = HDF5CreatePropertyList[H5PLINKCREATE]},
		If[FailureQ[plistHndl], Return[$Failed]];
		h5psetcreateintermediategroup[First[plistHndl], If[createIntermediateQ, 1, -1]];
		h5psetcharencoding[First[plistHndl], encoding];
		plistHndl
	]

DefaultLinkCreatePropList[] := CreateLinkPropList[True, H5TCSETUTF8];

HDF5CreateGroup[HDF5FileHandle[id_], path_?StringQ, HDF5PropListHandle[plistID_]] :=
	CheckHandle[HDF5GroupHandle, h5gcreate, id, path, plistID, H5PDEFAULT, H5PDEFAULT];

HDF5CreateGroup[HDF5FileHandle[id_], path_?StringQ] :=
	Block[{gid, defaultPropList = DefaultLinkCreatePropList[]},
		gid = HDF5CreateGroup[HDF5FileHandle[id], path, defaultPropList];
		h5pclose[defaultPropList];
		gid
	]

HDF5CreateGroup[___] := $Failed;

HDF5OpenGroup[HDF5FileHandle[id_], path_] := CheckHandle[HDF5GroupHandle, h5gopen, id, path];
	
HDF5CloseGroup[HDF5GroupHandle[groupId_]] := h5gclose[groupId];

HDF5ObjectType[HDF5FileHandle[gid_] | HDF5GroupHandle[gid_], linkName_String] :=
	Module[ { objectId, objectType },
		objectId = h5oopen[gid, linkName, H5PDEFAULT];
		objectType = If[Internal`NonNegativeIntegerQ[objectId],
			h5ogetinfotype[objectId]
			,
			$Failed
		];
		h5oclose[objectId]; 
		objectType
	]

HDF5ObjectType[fname_String, linkName_String] := openFileExecuteFAndClose[HDF5ObjectType, fname, linkName];
	
HDF5OpenObject[HDF5FileHandle[fileId_], path_String] := CheckHandle[HDF5ObjectHandle, h5oopen, fileId, path, H5PDEFAULT];
	
HDF5CloseObject[HDF5ObjectHandle[objectId_]] := h5oclose[objectId];

HDF5CreateDatatype[dtypeClass_Integer, size_Integer] := CheckHandle[HDF5DatatypeHandle, h5tcreate, dtypeClass, size];

HDF5OpenDatatype[HDF5FileHandle[fileId_], path_] := CheckHandle[HDF5DatatypeHandle, h5topen, fileId, path, H5PDEFAULT];

HDF5CloseDatatype[HDF5DatatypeHandle[typeId_]] := h5tclose[typeId]

HDF5CopyDatatype[HDF5DatatypeHandle[typeId_]] := CheckHandle[HDF5DatatypeHandle, h5tcopy, typeId];

HDF5CommitDatatype[HDF5FileHandle[fid_], path_?StringQ, HDF5DatatypeHandle[dtypeID_], HDF5PropListHandle[plistID_]] :=
	CheckHandle[HDF5DatatypeHandle, h5tcommit, fid, path, dtypeID, plistID, H5PDEFAULT, H5PDEFAULT];

HDF5CommitDatatype[fid_HDF5FileHandle, path_?StringQ, dtypeID_HDF5DatatypeHandle] :=
	Block[{comTypeID, defaultPropList = DefaultLinkCreatePropList[]},
		comTypeID = HDF5CommitDatatype[fid, path, dtypeID, defaultPropList];
		h5pclose[defaultPropList];
		comTypeID
	];

HDF5CreateTypeFromString[typeDescription_String] :=
	Replace[h5lttexttodtype[typeDescription], {id_?Internal`PositiveIntegerQ :> HDF5DatatypeHandle[id], _ :> $Failed}]

HDF5CommitTypeFromString[fid_HDF5FileHandle, dtypePath_String, typeDescription_String] :=
	Block[{dtypeID, plistID},
		dtypeID = h5lttexttodtype[typeDescription];
		If[FailedQ[dtypeID] || dtypeID < 1, (* There is a bug in H5LTtext_to_dtype so we have to check for 1 as well *)
			Return[$Failed]
		];
		HDF5CommitDatatype[fid, dtypePath, HDF5DatatypeHandle[dtypeID]]
	]


HDF5CreateDataset[HDF5FileHandle[id_], path_?StringQ, HDF5DatatypeHandle[dtypeID_], HDF5DataspaceHandle[dspaceID_], lcPropList_Integer, dsetCreatPropList_Integer] :=
	CheckHandle[HDF5DatasetHandle, h5dcreate, id, path, dtypeID, dspaceID, lcPropList, dsetCreatPropList, H5PDEFAULT];

HDF5CreateDataset[___] := $Failed;
	
HDF5OpenDataset[HDF5FileHandle[fileId_], path_] := CheckHandle[HDF5DatasetHandle, h5dopen, fileId, path, H5PDEFAULT];

HDF5CloseDataset[ HDF5DatasetHandle[datasetId_]] := h5dclose[datasetId]

HDF5DeleteLink[HDF5GroupHandle[gid_], linkName_?StringQ] := CheckTrue[h5ldelete, gid, linkName, H5PDEFAULT];

HDF5LinkExistsQ[HDF5GroupHandle[gid_], linkName_?StringQ] := Internal`PositiveIntegerQ @ h5lexists[gid, linkName, H5PDEFAULT];

HDF5CreateSoftLink[HDF5GroupHandle[gid_], linkName_String, targetPath_String, HDF5PropListHandle[plistID_]] :=
	CheckHDF5[h5lcreatesoft, targetPath, gid, linkName, plistID, H5PDEFAULT];

HDF5CreateHardLink[HDF5GroupHandle[gid_], linkName_String, targetPath_String, HDF5PropListHandle[plistID_]] :=
	CheckHDF5[h5lcreatehard, gid, targetPath, gid, linkName, plistID, H5PDEFAULT];
	
HDF5DatasetTypeString[HDF5DatasetHandle[datasetId_], verbosity_:1] :=
	Module[ {dtype, typeStr},
		dtype = h5dgettype[datasetId];
		typeStr = HDF5PrintType[dtype, verbosity];
		h5tclose[dtype];
		typeStr
	]
	
HDF5DataTypeIsComplex[HDF5DatasetHandle[datasetId_]] :=
	Module[ {dtype, res},
		dtype = h5dgettype[datasetId];
		res = h5tiscomplex[dtype];
		h5tclose[dtype];
		res
	]

HDF5RegisterComplexType[Automatic] := 
	Block[{tmp = h5tregistercomplex[True, False, "", ""]},
		If[FailedQ[tmp],
			False
			, 
			{H5TMCOMPLEX64, H5TMCOMPLEX128, H5TMCOMPLEX} = tmp;
			$UseComplexType = True;
			True
		]
	]
HDF5RegisterComplexType[None] :=
	Block[{tmp = h5tregistercomplex[False, True, "", ""]},
		If[FailedQ[tmp],
			False
			, 
			{H5TMCOMPLEX64, H5TMCOMPLEX128, H5TMCOMPLEX} = tmp;
			$UseComplexType = False;
			True
		]
	]
HDF5RegisterComplexType[{re_String, im_String}] := 
	If[re === im,
		HDF5RegisterComplexType[None]
		,
		Block[{tmp = h5tregistercomplex[False, False, re, im]},
			If[FailedQ[tmp], 
				False
				, 
				{H5TMCOMPLEX64, H5TMCOMPLEX128, H5TMCOMPLEX} = tmp;
				$UseComplexType = True;
				True
			]
		]
	];
HDF5UnregisterComplexType[] := HDF5RegisterComplexType[None];
	
HDF5DatasetDims[ HDF5DatasetHandle[datasetId_]] :=
	Module[ {dspaceId, dims},
		dspaceId = h5dgetspace[datasetId];
		dims = h5sgetsimpleextentdims[dspaceId];
		h5sclose[dspaceId];
		dims
	]

HDF5DatasetEncoding[ HDF5DatasetHandle[datasetId_]] :=
	Module[{plistId, nfilters, ret},
		plistId = h5dgetcreateplist[datasetId];
		nfilters = h5pgetnfilters[plistId];
		ret = Switch[nfilters,
			0, "None",
			1, HDF5GetFilterString[h5pgetfilterid[plistId, 0]],
			_, Table[HDF5GetFilterString[h5pgetfilterid[plistId, i]], {i, 0, nfilters-1}]
		]; 
		h5pclose[plistId]; 
		ret
	]

HDF5CreateDataspace[dims_, maxdims_] :=
    Block[{validDimQ = Function[d, Internal`PositiveMachineIntegerQ[d] || d == H5SUNLIMITED], dimLen = Length[dims]},
	    If[VectorQ[dims, validDimQ] && VectorQ[maxdims, validDimQ] && dimLen == Length[maxdims],
		    CheckHandle[HDF5DataspaceHandle, h5screatesimple, dimLen, dims, maxdims]
			,
			$Failed
		]
    ];
HDF5CreateDataspace[dims_] :=
	CheckHandle[HDF5DataspaceHandle, h5screatesimplen, Length[dims], dims] /; VectorQ[dims, (Internal`PositiveMachineIntegerQ[#] || # == H5SUNLIMITED)&];
HDF5CreateDataspace[0 | {}] := CheckHandle[HDF5DataspaceHandle, h5screate, H5SSCALAR];
HDF5CreateDataspace[0 | {}, _] := HDF5CreateDataspace[0];
HDF5CreateDataspace[_] := $Failed;

HDF5GetDataspaceDimensions[HDF5DataspaceHandle[dspaceID_]] :=
    Block[{dims = h5sgetsimpleextentdims[dspaceID]},
	    If[ListQ[dims],
		    dims
		    ,
		    $Failed
	    ]
    ];

HDF5CreateAttribute[HDF5ObjectHandle[objID_], attrName_?StringQ, HDF5DatatypeHandle[dtypeID_], HDF5DataspaceHandle[dspaceID_], lcPropList_Integer] :=
	CheckHandle[HDF5AttributeHandle, h5acreate, objID, attrName, dtypeID, dspaceID, lcPropList, H5PDEFAULT];

HDF5GetAttributeDataspace[HDF5AttributeHandle[attrID_]] := CheckHandle[HDF5DataspaceHandle, h5agetspace, attrID];

HDF5AttributeExists[HDF5ObjectHandle[objID_], attrName_?StringQ] :=
	Block[{status = h5aexists[objID, attrName]},
		Which[
			Internal`PositiveIntegerQ[status],
				True,
			status === 0,
				False,
			True,
				$Failed
		]
	];

HDF5DeleteAttribute[HDF5ObjectHandle[objID_], attrName_?StringQ] := CheckTrue[h5adelete, objID, attrName];

(* ::Subsubsection:: *)
(* Reading data *)

checkHyperslabParams[dims_, opt___] :=
	Module[{rank, offset, offsetOpt, pos, stride, strideOpt, rem, count, countOpt, block, blockOpt, len},
		rank = Length[dims];
		 
		offset = ConstantArray[0, rank];
		offsetOpt = "Offset" /. opt;
		If[offsetOpt =!= "Offset",
			If[IntegerQ[offsetOpt], offsetOpt = ConstantArray[offsetOpt, rank]];
			Which[
				VectorQ[offsetOpt, (IntegerQ[#] && NonNegative[#])&],
					offsetOpt = PadRight[offsetOpt, rank];
					pos = Flatten @ Position[MapThread[Less, {offsetOpt, dims}], False];
					If[pos =!= {},
						Message[Import::general, "Out-of-bound \"Offset\" value at positions "<> ToString[pos]<>", 0 will be used instead."];
						offsetOpt[[pos]] = 0;
					];
					offset = offsetOpt,
				offsetOpt =!= Automatic,
					Message[Import::general, "\"Offset\" should be a list of "<> ToString[rank] <>" non-negative integers; default \"Offset\" values will be used instead."];
			]
		];
		
		stride = ConstantArray[1, rank];
		strideOpt = "Stride" /. opt;
		If[strideOpt =!= "Stride",
			If[IntegerQ[strideOpt], strideOpt = ConstantArray[strideOpt, rank]];
			Which[
				VectorQ[strideOpt, (IntegerQ[#] && Positive[#])&],
					strideOpt = PadRight[strideOpt, rank, 1];
					pos = Flatten @ Position[MapThread[Less, {strideOpt, dims}], False];
					If[pos =!= {},
						Message[Import::general, "Out-of-bound \"Stride\" value at positions "<> ToString[pos]<>", 1 will be used instead."];
						strideOpt[[pos]] = 1;
					];
					stride = strideOpt,
				strideOpt =!= Automatic,
					Message[Import::general, "\"Stride\" should be a list of "<> ToString[rank] <>" positive integers; default \"Stride\" values will be used instead."];
			]
		];
		
		block = ConstantArray[1, rank];
		blockOpt = "Block" /. opt;
		If[blockOpt =!= "Block",
			If[IntegerQ[blockOpt], blockOpt = ConstantArray[blockOpt, rank]];
			Which[
				VectorQ[blockOpt, (IntegerQ[#] && Positive[#])&],
					blockOpt = PadRight[blockOpt, rank, 1];
					pos = Flatten @ Position[MapThread[LessEqual, {blockOpt, stride}], False];
					If[pos =!= {},
						Message[Import::general, "Out-of-bound \"Block\" value at positions "<> ToString[pos]<>", 1 will be used instead."];
						blockOpt[[pos]] = 1;
					];
					block = blockOpt,
				blockOpt =!= Automatic,
					Message[Import::general, "\"Block\" should be a list of "<> ToString[rank] <>" positive integers; default \"Block\" values will be used instead."];
			]
		]; 
		
		{count, rem} = Transpose @ QuotientRemainder[dims - offset, stride];
		count += Boole[MapThread[LessEqual, {block, rem}]];
		countOpt = "Count" /. opt;
		If[countOpt =!= "Count",
			If[IntegerQ[countOpt], countOpt = ConstantArray[countOpt, rank]];
			Which[
				VectorQ[countOpt, (IntegerQ[#] && Positive[#])&],
					len = Length[countOpt];
					countOpt = If[len >= rank,
						Take[countOpt, UpTo[rank]]
						,
						Join[countOpt, Take[count, {len + 1, rank}]]
					];
					pos = Flatten @ Position[MapThread[GreaterEqual, {dims, offset + (countOpt -1) * stride + block}], False];
					If[pos =!= {},	
						Message[Import::general, "Out-of-bound \"Count\" value at positions "<> ToString[pos]<>", default value will be used instead."];
						countOpt[[pos]] = count[[pos]];
					];
					count = countOpt,
				!MatchQ[countOpt, Automatic | All | "All"],
					Message[Import::general, "\"Count\" should be a list of "<> ToString[rank] <>" positive integers; default \"Count\" values will be used instead."];
			]
		];
		
		Return[{offset, stride, count, block}];	
	]

validPoints[ptsIn_, rank_, dims_] :=
	Block[{pts, maxCoords},
		If[!ArrayQ[ptsIn, 1|2, (IntegerQ[#] && Positive[#])&],
			Message[Import::general, "Points coordinates in \"TakeElements\"->" <> ToString[ptsIn] <> " should pe positive integers."];
			Return[False];
		];
		pts = If[VectorQ[ptsIn], Transpose[{ptsIn}], ptsIn];
		If[Dimensions[pts][[2]] != rank,
			Message[Import::general, "Expecting array with second dimension equal to dataset rank in \"TakeElements\"->" <> ToString[ptsIn]];
			Return[False];
		];
		maxCoords = Max /@ Transpose[pts];
		If[MemberQ[MapThread[LessEqual, {maxCoords, dims}], False],
			Message[Import::general, "Points coordinates should not exceed dataset dimensions in \"TakeElements\"->" <> ToString[ptsIn]];
			Return[False];
		];
		True
	]
	
getDataspaceSelection[fspace_, dims_, rank_, options__] :=
	Block[{selection, coords, numElems, memSpaceId, parseSel, allPos, dimsRepl, start, end, offset, stride, count, block},
		selection = "TakeElements" /. options /. {"TakeElements" -> All};
		
		Which[
			MemberQ[{Automatic, {}, All, "All"}, selection], 
				Return[H5SALL],
			
			ArrayQ[selection, 1|2, IntegerQ],
				If[!validPoints[selection, rank, dims], Return[$Failed]];
				If[VectorQ[selection], selection = Transpose[{selection}]];
				coords = NumericArray[selection - 1, "UnsignedInteger64"];
				numElems = Length[selection];
				If[NumericArrayQ[coords] && h5sselectelements[fspace, H5SSELECTSET, numElems, coords] >= 0,
					(* TODO: check if valid selection *)
					If[FailedQ[memSpaceId = h5screatesimplen[1, {numElems}]], Return[$Failed]];
					Return[memSpaceId];
					,
					Message[Import::general, "Cannot select points " <> ToString[selection]];
					Return[$Failed];
				],
				
			VectorQ[selection, MatchQ[#, Rule["Offset" | "Count" | "Block" | "Stride", _]] &],
				{offset, stride, count, block} = checkHyperslabParams[dims, selection];
				(* Select a hyperslab to read from the file data space *)
				h5sselecthyperslab[fspace, H5SSELECTSET, offset, stride, count, block];
				Return[h5screatesimplen[rank, count * block]],
			
			VectorQ[selection, (Head[#] === Span) || MatchQ[#, All|"All"] &],
				If[Length[selection] > rank,
					Message[Import::general, "Length of subset specification "<> ToString[selection] <>" should be no longer than the dataset rank."];
					Return[$Failed];
				];

				parseSel = selection /. {Span[a_, b_, c_] -> {a, b, c}, Span[a_, b_] -> {a, b, 1}, "All"|All -> {1, All, 1}};
				parseSel = PadRight[parseSel, rank, {{1, All, 1}}];
				If[Not[ArrayQ[parseSel, 2] && Dimensions[parseSel] === {rank, 3}],
					Message[Import::general, "Invalid subset specification "<> ToString[selection]];
					Return[$Failed];
				];
				parseSel = MapIndexed[
					Which[
						MatchQ[#1, All|"All"], dims[[First[#2]]],
						#1 === 0, Message[Import::general, "Invalid subset specification "<> ToString[selection]]; Return[$Failed],
						#1 < 0, dims[[First[#2]]] + #1 + 1,
						True, #1
					]&, parseSel, {2}];
				{start, end, stride} = Transpose[parseSel]; 
				count = Quotient[end - start, stride] + 1;
				If[MemberQ[count, n_/; n <= 0], Return[{}]];
				{offset, stride, count, block} = checkHyperslabParams[dims, {"Offset" -> start - 1, "Stride" -> stride, "Count" -> count}];

				(* Select a hyperslab to read from the file data space *)
				h5sselecthyperslab[fspace, H5SSELECTSET, offset, stride, count, block];
				Return[h5screatesimplen[rank, count * block]],
	
			True,
				Message[Import::general, "Value of \"TakeElements\" should be one of: All, Automatic, a list of point coordinates, a list of spans or a hyperslab specification"];
				Return[$Failed];
		]
	]
			
HDF5ReadDatasetRaw[HDF5DatasetHandle[datasetId_], opt___] :=
	Module[{fSpaceId, fTypeId, fclass, nTypeId, rank, dims, dupfSpaceId, memSpaceId, data},

		memSpaceId = H5SALL;
		(* get the data space of the dataset *)
		fSpaceId = h5dgetspace[datasetId];
		Switch[h5sgetsimpleextenttype[fSpaceId],
			H5SNULL,
				h5sclose[fSpaceId];
				Return[{}],
			H5SSCALAR,
				dupfSpaceId = H5SALL,
			H5SSIMPLE,
				(* get the rank and dimensions of the dataset from the HDF5 file data space *)
				rank = h5sgetsimpleextentndims[fSpaceId];
				dims = h5sgetsimpleextentdims[fSpaceId];
		
				(* duplicate file dataspace for subsetting *)
				dupfSpaceId = h5scopy[fSpaceId];
				h5sclose[fSpaceId];

				memSpaceId = getDataspaceSelection[dupfSpaceId, dims, rank, {opt}];

				If[memSpaceId === {}, Return[{}]];
				If[!IntegerQ[memSpaceId], Return[$Failed]],
			_,
				h5sclose[fSpaceId];
				Message[Import::general, "Invalid dataspace class."];
				Return[$Failed];
		];
		(* get the type of the data in the HDF5 file *)
		fTypeId = h5dgettype[datasetId];
		fclass = h5tgetclass[fTypeId];
		(* translate the file data type to the equivalent native memory data type *)
		nTypeId = h5tgetnativetype[fTypeId, H5TDIRASCEND];
		
		(* Read data, do not normalize NumericArrays *)
		data = h5dread[datasetId, nTypeId, memSpaceId, dupfSpaceId, H5PDEFAULT];
		
		(* clean up *)
		If[memSpaceId =!= H5SALL, h5sclose[memSpaceId]];
		If[dupfSpaceId =!= H5SALL, h5sclose[dupfSpaceId]];
		h5tclose /@ {nTypeId, fTypeId};
		
		If[FailedQ[data], $Failed, data]
	] (* End HDF5ReadDatasetRaw[ ] *)

HDF5ReadDataset = NormalizeNumericArray @* HDF5ReadDatasetRaw;

HDF5GetFilterString[filterId_] :=
	Block[{},
		Switch[filterId,
			H5ZFILTERERROR, "ERROR",
			H5ZFILTERNONE, "None",
			H5ZFILTERDEFLATE, "GZIP",
			H5ZFILTERSHUFFLE, "Shuffle",
			H5ZFILTERFLETCHER32, "Fletcher32",
			H5ZFILTERSZIP, "SZIP",
			H5ZFILTERNBIT, "N-Bit",
			H5ZFILTERSCALEOFFSET, "Scale+Offset",
			_, "Unknown(" <> ToString[filterId] <> ")"]
	]

HDF5GetStructuredDatasets[HDF5FileHandle[fileId_], groupName_String, lvl_:Infinity] :=
	Block[{groupHndl, ret, dsetCallb},
		If[FailureQ[groupHndl = HDF5OpenGroup[HDF5FileHandle[fileId], groupName]],
			Return[$Failed]
		];
		dsetCallb[dsetId_, args___] := HDF5ReadDataset[HDF5DatasetHandle[dsetId]];
		ret = First[listContents[groupHndl, dsetCallb, Null, lvl, groupName], $Failed];
		HDF5CloseGroup[groupHndl];
		ret
	]

HDF5GetGroupContents[HDF5FileHandle[fileId_], groupName_String, readData_:True, verb_:1, lvl_:Infinity] :=
	Block[{groupHndl, ret, dsetCallb, dtypeCallb},
		If[FailureQ[groupHndl = HDF5OpenGroup[HDF5FileHandle[fileId], groupName]],
			Return[$Failed]
		];
		If[readData,
			dsetCallb[dsetId_, args___] := HDF5ReadDataset[HDF5DatasetHandle[dsetId]]
			,
			dsetCallb[dsetId_, args___] := Association[{
						"DataFormat"	->	HDF5DatasetTypeString[HDF5DatasetHandle[dsetId], verb],
	 					"Dimensions"	->	HDF5DatasetDims[HDF5DatasetHandle[dsetId]],
	 					"DataEncoding"	->	HDF5DatasetEncoding[HDF5DatasetHandle[dsetId]]
					}]
		];
		dtypeCallb[dtypeId_, args___] := HDF5PrintType[dtypeId, verb];
		ret = First[listContents[groupHndl, dsetCallb, dtypeCallb, lvl, groupName], $Failed];
		HDF5CloseGroup[groupHndl];
		ret
	]

addPath[path_String, objName_String] := If[StringTake[path, -1] === "/", path <> objName, path <> "/" <> objName];

(* List the contents of a Group by link index *)
listContents[HDF5GroupHandle[gid_], dsetCallback_, dtypeCallback_, lvl_:Infinity, pathToGroup_String:".", visitedNodes_List:{}] :=
	Module[{groupAddr, contents, numLinks, linkName, linkObjId, objType, linkObjAddr, objName, nodes, datasetId, groupId, typeId},
		If[ FailedQ[numLinks = h5ggetinfonlinks[gid]] ||
			FailedQ[groupAddr = h5ogetinfoaddr[gid]],
				Return[{$Failed, visitedNodes}];
		];
		contents = Association[];
		nodes = visitedNodes;
		If[lvl <= 0, Return[{contents, nodes}]];
		If[FreeQ[nodes, groupAddr -> _],
			AppendTo[nodes, groupAddr -> pathToGroup]
		];
		Scan[(
			linkName = h5lgetnamebyidx[gid, ".", H5INDEXNAME, H5ITERINC, # - 1];
			objName = addPath[pathToGroup, linkName];
			(* Check for valid link *)
			If[h5oexistsbyname[gid, linkName, H5PDEFAULT] > 0,
				If[Not @ FailedQ[linkObjId = h5oopen[gid, linkName, H5PDEFAULT]],
					objType = h5ogetinfotype[linkObjId];
					linkObjAddr = h5ogetinfoaddr[linkObjId];
					h5oclose[linkObjId];
					If[FreeQ[nodes, linkObjAddr -> _],
						AppendTo[nodes, linkObjAddr -> objName];
						Switch[objType,
							H5OTYPEDATASET, (* open dataset and invoke proper callback function *)
								datasetId = h5dopen[gid, linkName, H5PDEFAULT];
								contents[linkName] = dsetCallback[datasetId];
						 		h5dclose[datasetId],
						 	H5OTYPENAMEDDATATYPE, (* open named datatype and invoke proper callback function *)
						 		If[dtypeCallback =!= Null,
									typeId = h5topen[gid, linkName, H5PDEFAULT]; 
									contents[linkName] = dtypeCallback[typeId];
									h5tclose[typeId]
						 		],
							H5OTYPEGROUP, (* Recursively list contents *)
								groupId = h5gopen[gid, linkName];
								{contents[linkName], nodes} = listContents[HDF5GroupHandle[groupId], dsetCallback, dtypeCallback, lvl-1, objName, nodes];
								h5gclose[groupId],
							_,
								contents[linkName] = HDF5UnknownObject[objName]
						];
						, 
						contents[linkName] = linkObjAddr /. nodes
					];
					,
					contents[linkName] = HDF5InaccessibleObject[objName]
				]
				,
				contents[linkName] = HDF5MissingObject[objName]
			])&, Range[numLinks]
		]; (* End Scan[] *)
		Return[{contents, nodes}];
	] (* End listContents[] *)


(* works for absolute paths returned by HDF5 *)
makePathAbsolute[path_String, loc_String:"/"] := StringReplace["/" <> loc <> If[path === ".", "", "/" <> path], "/" .. -> "/"];

objectDataFields = {"Location", "Name", "FileNumber", "Address", "Type", "NumAttrs"};
objectIndexRules = MapThread[#1 -> #2 &, {objectDataFields, Range[6]}];

HDF5GetObjectsData[fHandle_HDF5FileHandle, data__, startPath_String:"/", level_Integer:-1] := 
	Module[{locId, fields, fldIndcs, trav, objData},
		locId = First @ fHandle;
		fields = Select[data, MemberQ[objectDataFields, #]&];
		fldIndcs = fields /. objectIndexRules;
		trav = If[level >= 0, 
			h5traversetolevel[locId, level, startPath]
			,
			h5traverse[locId, 0, startPath]
		];

		If[FailedQ[trav],
			trav = {}
		];
		objData = Part[#, fldIndcs]& /@ trav;
		Return[objData];
	]
	
HDF5GetObjectsData[fileName_String, params___] := openFileExecuteFAndClose[HDF5GetObjectsData, fileName, params];

HDF5GetAttributes[fileName_String] :=
	Module[{objData, attrs},
		If[FailureQ[HDF5ToolsInit[]], Return[$Failed]];
		If[FailureQ[objData = HDF5GetObjectsData[fileName, {"Name", "NumAttrs"}]], 
			Return[$Failed]
		];
		attrs = MapThread[#1 -> #2 &, {makePathAbsolute /@ objData[[All, 1]], HDF5GetAttributes[fileName, Sequence @@ #] & /@ objData}];
 		Return[attrs];
	] 
	
HDF5GetAttributes[fileName_String, objectName_String, objectNumAttr_Integer:-1] :=
	Module[{fHandle, oHandle, objectId, numAttrs, attrs, attrName, attrId, attrTypeId, attrVal},
		If[FailureQ[HDF5ToolsInit[]], Return[$Failed]];
		fHandle = HDF5OpenFile[fileName, H5FACCRDONLY];
		If[FailureQ[fHandle], Return[$Failed]];
		oHandle = HDF5OpenObject[fHandle, objectName];
		If[FailureQ[oHandle], Return[$Failed]];
		objectId = First @ oHandle;
		numAttrs = If[objectNumAttr >= 0, objectNumAttr, h5ogetinfonumattrs[objectId]];
		attrs = Association @ Table[
 				attrName = h5agetnamebyidx[objectId, ".", H5INDEXNAME, H5ITERINC, index];
 				attrId = h5aopen[objectId, attrName];
 				attrTypeId = h5agettype[attrId];
 				attrVal = NormalizeNumericArray @ h5aread[attrId, attrTypeId];
 				h5tclose[attrTypeId];
 				h5aclose[attrId]; 
 				attrName -> attrVal,
 				 {index, 0, numAttrs - 1}];
 		h5oclose[objectId];
 		HDF5CloseFile[fHandle];
 		Return[attrs];
	]

countMissingObjects[HDF5FileHandle[fileId_]] := 
	Block[{lnks, softLnks},
		If[FailedQ[lnks = h5traverse[fileId, 1, "/"]], Return[0]];
		softLnks = Select[lnks, (Length[#] === 6 && #[[3]] === H5LTYPESOFT && (h5oexistsbyname[fileId, #[[6]], H5PDEFAULT] <= 0))&];
		Return[Length[softLnks]];
	]
	
HDF5GetSummary[fileName_String] :=
	Block[{res, fHandle, fFormat, fSize, fSizeQuan, objData, objCnt, namedTypesCount, missingCount},
		res = <||>;
		If[FailureQ[fHandle = HDF5OpenFile[fileName, H5FACCRDONLY]], 
			fFormat = FileFormat[fileName];
			res["Format"] = If[fFormat === "HDF5", "HDF5 (broken)", fFormat];
			Return[Dataset @ res]
			,
			res["Format"] = "HDF5";
		];
		fSize = h5fgetfilesize[First[fHandle]];
		objData = HDF5GetObjectsData[fHandle, {"Type"}];
		missingCount = countMissingObjects[fHandle];
		HDF5CloseFile[fHandle];
		If[FailureQ[objData], Return[Dataset @ res]];
		objCnt = Counts[Flatten[objData]];
		res["GroupCount"] = Lookup[objCnt, H5OTYPEGROUP, 0];
		res["DatasetCount"] = Lookup[objCnt, H5OTYPEDATASET, 0];
		namedTypesCount = Lookup[objCnt, H5OTYPENAMEDDATATYPE, 0];
		If[namedTypesCount > 0, res["NamedDatatypeCount"] = namedTypesCount];
		If[missingCount > 0, res["MissingObjectCount"] = missingCount];
		res["FileName"] = FileNameTake[fileName];
		If[!FailedQ[fSize],
			fSizeQuan = Quantity[fSize, "Bytes"];
			res["FileSize"] = Which[
				10^3 <= fSize < 10^6, N[UnitConvert[fSizeQuan, "Kilobytes"], 3],
				10^6 <= fSize < 10^9, N[UnitConvert[fSizeQuan, "Megabytes"], 3],
				10^9 <= fSize , N[UnitConvert[fSizeQuan, "Gigabytes"], 3],
				True, fSizeQuan
			];
		];
		Return[Dataset @ res];
	]
	
HDF5GetExtraInfo[fileName_String] :=
	Block[{fHandle, objData, meta},
		If[FailureQ[fHandle = HDF5OpenFile[fileName, H5FACCRDONLY]], Return[$Failed]];
		If[FailureQ[objData = HDF5GetObjectsData[fHandle, {"Name", "Type", "NumAttrs"}]],
			HDF5CloseFile[fHandle];
			Return[$Failed]
		];
		meta = MapThread[#1 -> #2 &, {makePathAbsolute /@ objData[[All, 1]], HDF5GetExtraInfo[fHandle, Sequence @@ #] & /@ objData}];
		HDF5CloseFile[fHandle];
 		Return[meta];
	]

HDF5GetExtraInfo[fileName_String, obj_String] := 
	Block[{fHandle, rootId, objId, meta},
		If[FailureQ[fHandle = HDF5OpenFile[fileName, H5FACCRDONLY]], Return[$Failed]];
		rootId = First[fHandle];
		meta = If[h5oexistsbyname[rootId, obj, H5PDEFAULT] > 0,
			If[Not @ FailedQ[objId = h5oopen[rootId, obj, H5PDEFAULT]],
				HDF5GetExtraInfo[fHandle, obj, h5ogetinfotype[objId], h5ogetinfonumattrs[objId]]
				,
				Association[{"Type" -> "InaccessibleObject"}]
			]
			,
			$Failed
		];
		HDF5CloseFile[fHandle];
		Return[meta];
	]
	
HDF5GetExtraInfo[fHandle_HDF5FileHandle, objName_String, objType_Integer, objNumAttr_Integer] :=
	Module[{dHandle, tHandle, gHandle, meta},
		Switch[objType,
			H5OTYPEDATASET, (* open dataset and invoke proper callback function *)
				dHandle = HDF5OpenDataset[fHandle, objName];
				meta = Association[{
					"Type"			-> "Dataset",
					"AttributeCount"-> objNumAttr,
					"DataFormat"	-> HDF5DatasetTypeString[dHandle, Infinity],
 					"Dimensions"	-> HDF5DatasetDims[dHandle],
 					"DataEncoding"	-> HDF5DatasetEncoding[dHandle]
				}];
		 		HDF5CloseDataset[dHandle],
		 	H5OTYPENAMEDDATATYPE, (* open named datatype and invoke proper callback function *)
		 		tHandle = HDF5OpenDatatype[fHandle, objName];
				meta = Association[{
					"Type"			-> "Datatype",
					"AttributeCount"-> objNumAttr
				}];
				HDF5CloseDatatype[tHandle],
			H5OTYPEGROUP, (* Recursively list contents *)
				gHandle = HDF5OpenGroup[fHandle, objName];
				meta = Association[{
					"Type"			-> "Group",
					"AttributeCount"-> objNumAttr,
					"LinkCount"		-> h5ggetinfonlinks[First[gHandle]]
				}];
				HDF5CloseGroup[gHandle],
			_,
				meta = Association[{"Type" -> "UnknownType"}]
		];
 		Return[meta];
	]
		
HDF5GetGroupNames[fileName_String, group_String:"/"] :=
	Module[{objData},
		If[FailureQ[HDF5ToolsInit[]], Return[$Failed]];
		If[FailureQ[objData = HDF5GetObjectsData[fileName, {"Name", "Type"}, group]], 
			Return[$Failed]
		];
		Cases[objData, {a_, H5OTYPEGROUP} :> If[group === "/", makePathAbsolute[a], a]]
	] 

HDF5GetDatasetNames[f_, group_String:"/", returnFullPaths_:True, level_Integer:-1] :=
	Module[{startPath, objData, parsePath},
		startPath = makePathAbsolute[group];
		If[FailureQ[objData = HDF5GetObjectsData[f, {"Name", "Type"}, startPath, level]], 
			Return[$Failed]
		];
		parsePath[path_] := Which[
			returnFullPaths && level < 0,
				makePathAbsolute[path, group],
			!returnFullPaths && level >= 0,
				If[StringStartsQ[path, startPath], StringDrop[path, If[startPath === "/", 1, 1 + StringLength[startPath]]], path],
			True,
				path
		];
		Cases[objData, {a_, H5OTYPEDATASET} :> parsePath[a]]
	]

openFileExecuteFAndClose[f_, fileName_String, args___] := 
	Module[{fHandle, ret},
		If[FailureQ[fHandle = HDF5OpenFile[fileName, H5FACCRDONLY]], 
			Return[$Failed]
		];
		ret = f[fHandle, args];
		HDF5CloseFile[fHandle];
		Return[ret];
	]

(* ::Subsubsection:: *)
(* Printing datatype info *)	
	
simplePlural[word_String, 1] := word;
simplePlural[word_String, n_Integer] := word <> "s";

classToString[typeId_Integer] :=
	Switch[h5tgetclass[typeId],
		H5TNOCLASS,		"H5T_NO_CLASS (invalid datatype identifier)",	
		H5TINTEGER,		"H5T_INTEGER",
		H5TFLOAT,		"H5T_FLOAT",
		H5TSTRING,		"H5T_STRING",
		H5TBITFIELD,	"H5T_BITFIELD",
		H5TOPAQUE,		"H5T_OPAQUE",
		H5TCOMPOUND,	"H5T_COMPOUND",
		H5TREFERENCE,	"H5T_REREFERENCE",
		H5TENUM,		"H5T_ENUM",
		H5TVLEN,		"H5T_VLEN",
		H5TARRAY,		"H5T_ARRAY"
	]
		
stringInfoToString[typeId_Integer, indent_Integer] :=
	Module[{indStr, resStr},
		indStr = StringJoin @ Table["\t", indent];
		resStr = {indStr, "Character set:\t"};
		AppendTo[resStr, Switch[h5tgetcset[typeId], 
			H5TCSETUTF8, "UTF8\n",
			H5TCSETASCII, "US ASCII\n",
			_, "Error while checking character set.\n"]
		];
		AppendTo[resStr, indStr <> "Padding:\t"]; 
		AppendTo[resStr, Switch[h5tgetstrpad[typeId], 
			H5TSTRNULLTERM, "Null-terminated.\n",
			H5TSTRNULLPAD, "Padded with zeros.\n",
			H5TSTRSPACEPAD, "Padded with spaces.\n",
			_, "Error while checking padding type.\n"]
		];
		AppendTo[resStr, indStr <> Which[ 
			# == 0, "Fixed length string.",
			# > 0, "Variable length string.",
			True, "Error while checking string length."] & @ h5tisvariablestr[typeId]	
		];
		Return @ StringJoin[resStr];
	]
	
compoundTypeAssoc[typeId_Integer, level_] := 
	Block[{nmembers = h5tgetnmembers[typeId]},
		Association @ Map[
			Block[{memType = h5tgetmembertype[typeId, #-1], res}, 
				res = h5tgetmembername[typeId, #-1] -> HDF5PrintType[memType, level];
				h5tclose[memType]; 
				res
			]&, Range[nmembers]
		]
	]

compoundInfoToString[typeId_Integer, indent_Integer] := 
	Module[{indStr, resStr, nmembers, membersStr, i},
		indStr = StringJoin @ Table["\t", indent];
		nmembers = h5tgetnmembers[typeId];
		resStr = {indStr, "Compound type with " <> ToString @ nmembers, " ", simplePlural["member", nmembers], ":"};
		membersStr = Table[
			Block[{memType = h5tgetmembertype[typeId, i-1], str}, 
				str = StringJoin[
					"\n", indStr, ToString[i], ". \"", h5tgetmembername[typeId, i-1], "\":",
					"\n", indStr, "\tOffset:\t", ToString @ h5tgetmemberoffset[typeId, i-1],
					"\n", HDF5PrintFullTypeInfo[memType, indent + 1]
				];
				h5tclose[memType]; 
				str
			],
			{i, nmembers}];
		resStr = resStr ~Join~ membersStr;
		Return @ StringJoin[resStr];
	]
	
enumTypeAssoc[typeId_Integer] := 
	Block[{nmembers = h5tgetnmembers[typeId]},
		Association[h5tgetmembervalue[typeId, #-1] -> h5tgetmembername[typeId, #-1] & /@ Range[nmembers]]
	]
		
enumInfoToString[typeId_Integer, indent_Integer] := 
	Module[{indStr, resStr, nmembers, membersStr, btype, membValues},
		indStr = StringJoin @ Table["\t", indent];
		nmembers = h5tgetnmembers[typeId];
		resStr = {indStr, "Enumeration type with " <> ToString @ nmembers, " ", simplePlural["member", nmembers], ":"};
		btype = h5tgetsuper[typeId];
		membValues = h5tgetmembervalue[typeId, # - 1] & /@ Range[nmembers];
		membersStr = Table[
			StringJoin["\n", indStr, ToString[i], ". \"", h5tgetmembername[typeId, i-1], "\": ", ToString[membValues[[i]]]],
			{i, nmembers}];
		resStr = resStr ~Join~ membersStr;
		
		AppendTo[resStr, StringJoin["\n", indStr, "Base type:\n", HDF5PrintFullTypeInfo[btype, indent + 1]]];
		h5tclose[btype];
		Return @ StringJoin[resStr];
	]

arrayInfoToString[typeId_Integer, indent_Integer] :=
	Module[{indStr, resStr, baseType, rank, dims},
		indStr = StringJoin @ Table["\t", indent];
		rank = h5tgetarrayndims[typeId];
		dims = h5tgetarraydims[typeId];
		baseType = h5tgetsuper[typeId];
		resStr = {indStr, "Array type with rank " <> ToString @ rank, " and dimensions ", ToString @ dims, 
			".\n", indStr, "Base type:\n", HDF5PrintFullTypeInfo[baseType, indent + 1]};
		h5tclose[baseType]; 
		Return @ StringJoin[resStr];
	]
	
HDF5PrintType[typeId_Integer, 0] :=
	Block[{},
		If[FailureQ[HDF5ToolsInit[]], Return[$Failed]]; 
		If[h5tiscomplex[typeId],
			"Complex"
			,
			Switch[h5tgetclass[typeId],
				H5TNOCLASS,		"InvalidDatatype",	
				H5TINTEGER,		h5tgetnamestring[typeId],
				H5TFLOAT,		h5tgetnamestring[typeId],
				H5TSTRING,		"String",
				H5TBITFIELD,	"Bitfield",
				H5TOPAQUE,		"ByteArray",
				H5TCOMPOUND,	"Compound",
				H5TREFERENCE,	"Reference",
				H5TENUM,		"Enumerated",
				H5TVLEN,		"VariableLength",
				H5TARRAY,		"Array",
				_, 				"NotSupported"
			]
		]
	]
	
HDF5PrintType[typeId_Integer, verbosity_:2] :=
	Module[{class, newVer, ret},	
		If[FailureQ[HDF5ToolsInit[]], Return[$Failed]];
		newVer = verbosity - 1;
		class = h5tgetclass[typeId];
		Which[
			h5tiscomplex64[typeId],
				"ComplexReal32",
			h5tiscomplex128[typeId],
				"ComplexReal64",
			True,
				Switch[class,
					H5TNOCLASS,		"InvalidDatatype",	
					H5TINTEGER,		h5tgetnamestring[typeId],
					H5TFLOAT,		h5tgetnamestring[typeId],
					H5TSTRING,		"String",
					H5TBITFIELD,	"Bitfield",
					H5TOPAQUE,		Association["Class" -> HDF5PrintType[typeId, 0], "Length" -> h5tgetsize[typeId], "Tag" -> h5tgettag[typeId]],
					H5TCOMPOUND,	Association["Class" -> HDF5PrintType[typeId, 0], "Structure" -> compoundTypeAssoc[typeId, newVer]],
					H5TREFERENCE,	"Reference",
					H5TENUM,		Block[{s = h5tgetsuper[typeId]},
										ret = <|"Class" -> HDF5PrintType[typeId, 0], "DataFormat" -> HDF5PrintType[s, newVer], "Structure" -> enumTypeAssoc[typeId]|>;
										h5tclose[s];
										ret
									],
					H5TVLEN,		"VariableLength",
					H5TARRAY,		Block[{s = h5tgetsuper[typeId]},
										ret = <|"Class" -> HDF5PrintType[typeId, 0], "Dimensions" -> h5tgetarraydims[typeId], "DataFormat" -> HDF5PrintType[s, newVer]|>;
										h5tclose[s];
										ret
									],
					_, 				"NotSupported"
				]
		]	
	]

HDF5PrintType[HDF5DatatypeHandle[typeId_Integer], "Debug"] := HDF5PrintFullTypeInfo[typeId];
HDF5PrintType[typeId_Integer, "Debug"] := HDF5PrintFullTypeInfo[typeId];
		
HDF5PrintFullTypeInfo[typeId_Integer, indent_Integer:0] := 
	Module[{indStr, resStr, indx, size, class},	
		If[FailureQ[HDF5ToolsInit[]], Return[$Failed]];
		class = h5tgetclass[typeId];
		indStr = StringJoin @ Table["\t", indent];
		indx = Position[h5tequal[typeId, #] & /@ h5builtintypes, True];
		If[Length[indx] > 0,
			Return[indStr <> "Built-in datatype: {" <> StringRiffle[h5builtintypenames[[Flatten[indx]]], ", "] <> "}"]
		];
		size = h5tgetsize[typeId];
		resStr = {(*indStr, StringPadRight["Datatype id:", 15], ToString[typeId], "\n", *)
			indStr, StringPadRight["Class:", 15], classToString[typeId], "\n",
			indStr, StringPadRight["Size:", 15], ToString[size], " ", simplePlural["byte", size]};
		Switch[class,
			H5TSTRING, AppendTo[resStr, "\n" <> stringInfoToString[typeId, indent]],
			H5TCOMPOUND, AppendTo[resStr, "\n" <> compoundInfoToString[typeId, indent]],
			H5TENUM, AppendTo[resStr, "\n" <> enumInfoToString[typeId, indent]],
			H5TARRAY, AppendTo[resStr, "\n" <> arrayInfoToString[typeId, indent]]
		];
		Return @ StringJoin[resStr];	
	]
(* ::Subsubsection:: *)

HDF5GetTypeSize[memtype_Integer] :=
	Block[{retSize = h5tgetsize[memtype]},
		If[retSize == 0,
			Message[Export::general, ErrUnsuppDataType[memtype]];
			$Failed
			,
			retSize
		]
	]
HDF5GetTypeSize[HDF5DatatypeHandle[memtype_Integer]] := HDF5GetTypeSize[memtype]
HDF5GetTypeSize[_] := $Failed

HDF5NumericArrayTypeToMemoryType[raType_String] :=
	Switch[raType,
		"Integer8",				H5TNATIVEINT8,
		"UnsignedInteger8",		H5TNATIVEUINT8,
		"Integer16",			H5TNATIVEINT16,	
		"UnsignedInteger16",	H5TNATIVEUINT16,
		"Integer32",			H5TNATIVEINT32,
		"UnsignedInteger32",	H5TNATIVEUINT32,
		"Integer64",			H5TNATIVEINT64,
		"UnsignedInteger64",	H5TNATIVEUINT64,
		"Real32",				H5TMREAL32,
		"Real64",				H5TMREAL64,
		"ComplexReal32",		H5TMCOMPLEX64,
		"ComplexReal64",		H5TMCOMPLEX128,
		_, (Message[Export::general, ErrNumericArrayType[raType]]; $Failed)
	];

HDF5NumericArrayTypeToMemoryType[typeId_Integer] := typeId;
	
HDF5CreateCompound[assoc_?AssociationQ] :=
	Module[{compType, keys, types, sizes, offsets},
		If[FailureQ[HDF5ToolsInit[]], Return[$Failed]];
		keys = Keys[assoc];
		If[!Developer`StringVectorQ[keys],
			Message[Export::general, ErrFieldNames[assoc]];
			Return[$Failed]
		];
		types = HDF5NumericArrayTypeToMemoryType /@ Values[assoc];
		sizes = HDF5GetTypeSize /@ types;
		If[MemberQ[sizes, $Failed], 
			Return[$Failed]
		];
		offsets = FoldList[Plus, 0, sizes];
		compType = h5tcreate[H5TCOMPOUND, Last[offsets]];
		MapThread[h5tinsert[compType, #1, #2, #3] &, {keys, Drop[offsets, -1], types}];
		Return[compType];
	]


HDF5CreateEnum[baseType_Integer, enums_Association] := HDF5CreateEnum[baseType, Normal @ enums]


HDF5CreateEnum[baseType_Integer, enums : {__Rule}] :=
	Module[{enumType, keys, vals},
		If[FailureQ[HDF5ToolsInit[]], Return[$Failed]];
		keys = Keys[enums];
		If[!Developer`StringVectorQ[keys],
			Message[Export::general, ErrFieldNames[enums]];
			Return[$Failed]
		];
		vals = Values[enums];
		enumType = h5tenumcreate[baseType];
		MapThread[h5tenuminsert[enumType, #1, #2] &, {keys, vals}];
		Return[enumType];
	]

(* ::Subsubsection:: *)
(* Structure Graph *)

graphElemsColors = AssociationThread[
	{"Root Group", "Group", "Dataset", "Named Datatype", "Unknown/Missing object", "Symbolic Link", "Hard Link"},
	{Hue[.6, .4, 1], Hue[.6, 1, 1], Hue[.6, 1, .7], ColorData[106, 2], Lighter[Red, .8], GrayLevel[.8, 1], GrayLevel[.5, 1]}
];

graphVertShapes = Association[
	"Root Group" -> "Circle",
	"Group" -> "Circle",
	"Dataset" -> "Square", 
	"Named Datatype" ->  "Triangle",
	"Unknown/Missing object" -> Function[{p}, {Polygon[p + .075 # & /@ CirclePoints[8]], Text[Style["?", White, 18], p]}]
];

objectToVertex[{_, name_String, _, address_Integer, type_Integer, attrCnt_Integer}, rootAddr_Integer, startAddr_Integer] := 
	Module[{ret},
		formatAttributeString[0] = "";
		formatAttributeString[1] = " with 1 attribute";
		formatAttributeString[n_] := " with " <> ToString[n] <> " attributes";
		ret = Switch[type,
			H5OTYPEGROUP,
				If[address === rootAddr,
					{
						Tooltip[address, "Root Group" <> formatAttributeString[attrCnt]],
						address -> graphElemsColors["Root Group"],		(* VertexStyle 		*)
						address -> graphVertShapes["Root Group"], 		(* VertexShape 		*)
						address -> "",									(* VertexLabels	    *)
						address -> Directive[Bold, 15]					(* VertexLabelStyle *)
					}
					,
					{
						Tooltip[address, "Group" <> formatAttributeString[attrCnt]],
						address -> graphElemsColors["Group"],
						address -> graphVertShapes["Group"],
						address -> "",
						address -> Automatic
					}
				],
			H5OTYPEDATASET,
				{
					Tooltip[address, "Dataset" <> formatAttributeString[attrCnt]],
					address -> graphElemsColors["Dataset"], 
					address -> graphVertShapes["Dataset"],
					address -> "",
					address -> Automatic
				},
			H5OTYPENAMEDDATATYPE,
				{
					Tooltip[address, "Data type" <> formatAttributeString[attrCnt]],
					address -> graphElemsColors["Named Datatype"],
					address -> graphVertShapes["Named Datatype"],
					address -> "",
					address -> Automatic
				},
			H5OTYPEUNKNOWN,
				{
					Tooltip[address, "Unknown object" <> formatAttributeString[attrCnt]],
					address -> graphElemsColors["Unknown/Missing object"],
					address -> graphVertShapes["Unknown/Missing object"],
					address -> None,
					address -> Automatic
				},
			_, $Failed
		];
		(* Check if starting point, change style only if it's a group, otherwise the graph will be a single vertex *)
		If[address === startAddr && type == H5OTYPEGROUP, 
			ret[[3, 2]] = Function[{p}, {Disk[p, .07], Thick, Circle[p, .09]}];
		];
		PrependTo[ret, name -> address];
		Return[ret];
	]
objectToVertex[___] := $Failed;

graphLabel[lbl_String] := " " <> lbl <> " ";

splitPath[path_String] := 
	Block[{pathChunks, name, location},
		pathChunks = StringSplit[path, RegularExpression["\\/+"]];
		name = Last[pathChunks];
		pathChunks = Drop[pathChunks, -1];
		location = If[Length[pathChunks] === 0, ".", StringRiffle[pathChunks, "/"]];
		Return[{location, name}];
	];

hardLinkToEdge[{addrMap_, edges_}, {_, name_String, H5LTYPEHARD, cset_Integer, address_Integer, 0}] :=
	Module[{label, baseName, baseAddr, edge, e},
		{baseName, label} = splitPath[name];
		baseAddr = baseName /. addrMap;
		If[baseAddr === baseName, Return[{addrMap, edges}]];
		e = DirectedEdge[baseAddr, address];
		edge = {
			e,
			Tooltip[e, label],
			e -> Text[graphLabel[label], Background -> Lighter[GrayLevel[.7, 1], .5]],
			e -> graphElemsColors["Hard Link"]
		};
		Return[{If[MemberQ[addrMap, name -> address], addrMap, Append[addrMap, name -> address]], Append[edges, edge]}];
	];

hardLinkToEdge[x_, _] := x;

softLinkToEdge[{_, name_String, H5LTYPESOFT, cset_Integer, _, value_String}, addrMap_, fileId_, startPath_] := 
	Module[{linkName, linkLoc, baseAddr, targetAddr, edge, e},
		{linkLoc, linkName} = splitPath[startPath <> "/" <> name];
		baseAddr = linkName /. addrMap;
		If[baseAddr === linkName, baseAddr = h5ogetinfobynameaddr[fileId, linkLoc]];
		If[FailedQ[baseAddr], Return[$Failed]];
		If[h5oexistsbyname[fileId, value, H5PDEFAULT] > 0,
			targetAddr = h5ogetinfobynameaddr[fileId, value];
			e = UndirectedEdge[baseAddr, targetAddr];
			,
			e = UndirectedEdge[baseAddr, -1];
		];
		edge = {
			e,
			Tooltip[e, linkName],
			e -> Text[graphLabel[linkName], Background -> Lighter[GrayLevel[.7, 1], .5]],
			e -> Directive[graphElemsColors["Symbolic Link"], Dashing[0.005]]
		};
		Return[edge];
	];
softLinkToEdge[___] := $Failed; 

makeSoftEdgeFun[fromCoord_, toCoord_][pts_, e_] := 
		(* If there is something wrong with the edge, return Automatic *)
		If[pts === {} || MissingQ[fromCoord] || MissingQ[toCoord], 
			Return[Automatic]
			,			
			(* Check which vertex is closer to the first point of the edge *)
			If[Nearest[{fromCoord, toCoord} -> Automatic, pts[[1]], 1] === {1},
				 {Arrowheads[{{0.03, 0.8}}], Arrow @ pts}
				 ,
				 {Arrowheads[{{0.03, 0.8}}], Arrow @ Reverse @ pts}
			]
		]

HDF5StructureGraph[HDF5FileHandle[fileId_], startPath_String:"/"] := 
	Module[{objs, rootAddr,	startAddr, tmp, addrMap, vertices, vCoords, vStyle, vShapes, vLabels, vLabelsStyle, lnks, rawHardEdges, styledHardEdges, hardELabels, hardEStyle,
		rawSoftEdges, styledSoftEdges, eLabels, eStyle, hasMissing, graph},
		
		(* Traverse all objects in the file and create vertices *)
		If[FailedQ[objs = HDF5GetObjectsData[HDF5FileHandle[fileId], objectDataFields, startPath]], Return[$Failed]];
		rootAddr = h5ogetinfobynameaddr[fileId, "/"];
		startAddr = If[MatchQ[startPath, "/"|"."],
			rootAddr
			,
			h5ogetinfobynameaddr[fileId, startPath]
		];
		tmp = objectToVertex[#, rootAddr, startAddr]& /@ objs;
		If[tmp === {} || MemberQ[Flatten[tmp], $Failed], Return[$Failed]];
		{addrMap, vertices, vStyle, vShapes, vLabels, vLabelsStyle} = Transpose[tmp];

		(* Traverse all links in the file and create edges (hard and soft separately) *)
		lnks = h5traverse[fileId, 1, startPath];
		{addrMap, tmp} = Fold[hardLinkToEdge, {addrMap, {}}, lnks];
		{rawHardEdges, styledHardEdges, hardELabels, hardEStyle} = If[Length[tmp] == 0,
			{{}, {}, {}, {}}
			,
			Transpose @ tmp
		];
		tmp = DeleteCases[softLinkToEdge[#, addrMap, fileId, startPath]& /@ lnks, $Failed];
		{rawSoftEdges, styledSoftEdges, eLabels, eStyle} = If[Length[tmp] == 0,
			{{}, {}, {}, {}}
			,
			Transpose @ tmp
		];

		(* Set style for special vertex representing missing objects *)
		hasMissing = MemberQ[rawSoftEdges, UndirectedEdge[_, -1]];
		If[hasMissing, AppendTo[vStyle,  -1 -> graphElemsColors["Unknown/Missing object"]]];
		If[hasMissing, AppendTo[vShapes, -1 -> graphVertShapes["Unknown/Missing object"]]];
		If[hasMissing, AppendTo[vLabels, -1 -> None]];
			
		(* Create graph *)
		graph = Graph[
			vertices, styledHardEdges ~ Join ~ styledSoftEdges,
			VertexStyle -> vStyle,
			VertexShapeFunction -> vShapes,
			VertexLabels -> vLabels,
			VertexLabelStyle -> vLabelsStyle,
			VertexSize -> 0.2,
			EdgeLabels -> hardELabels~Join~eLabels,
			EdgeShapeFunction -> GraphElementData[{"ShortFilledArrow", "ArrowSize" -> 0.03}],
			EdgeStyle -> hardEStyle~Join~eStyle,
			GraphLayout -> { "LayeredEmbedding", "RootVertex" -> startAddr },    (* TODO: change to {"LayeredDigraphEmbedding", "RootVertex" -> startAddr}, *)
			GraphStyle -> "DiagramGold",                                        (* or make StructureGraphs customizable via options *)
			BaselinePosition -> Axis,
			FormatType -> TraditionalForm
		];

		(* Set proper style for soft links *)
		If[rawSoftEdges =!= {},
			vertices = VertexList[graph];
			vCoords = AssociationMap[PropertyValue[{graph, #}, VertexCoordinates]&, vertices];
			graph = Fold[SetProperty[{#1, #2}, EdgeShapeFunction -> makeSoftEdgeFun[vCoords[First[#2]], vCoords[Last[#2]]]]&, graph, rawSoftEdges];
		];		

		Return[graph];
	]

HDF5StructureGraphLegend = 
	SwatchLegend[
		Values[graphElemsColors], 
		
		(" - " <> #)& /@ Keys[graphElemsColors],
		
		LegendMarkers -> Graphics/@ {Disk[], Disk[], Polygon[CirclePoints[4]], Polygon[CirclePoints[3]], {Polygon[CirclePoints[8]], Text[Style["?", White, 18]]}, 
			{Dashed	, Line[{{0, 0}, {1, 0}}]}, Line[{{0, 0}, {1, 0}}]},
			
		LegendMarkerSize -> 30
	]

End[]

EndPackage[]
