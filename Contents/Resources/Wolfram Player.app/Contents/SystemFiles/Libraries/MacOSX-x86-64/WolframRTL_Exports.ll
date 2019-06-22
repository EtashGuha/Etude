; ModuleID = '/Volumes/Jenkins/workspace/Component.LLVM.RuntimeLibrary.MacOSX-x86-64.12.0/scratch/rtl_makefiles/lib/WolframRTL_Exports.bc'
source_filename = "/Volumes/Jenkins/workspace/Component.LLVM.RuntimeLibrary.MacOSX-x86-64.12.0/scratch/rtl_makefiles/lib/WolframRTL_Exports.bc"

%struct.st_ZSpan = type { %struct.st_ZRow*, %struct.st_ZSpan*, %struct.st_ZSpan*, %struct.st_ZSpan*, %struct.st_ZBlock**, %struct.st_ZBlock**, %struct.st_ZBlock*, i64 }
%struct.st_ZRow = type { %struct.st_ZAllocator*, %struct.st_KMutex*, %struct.st_ZSpan*, %struct.st_ZSpan*, %struct.st_ZSpan*, %struct.st_ZSpan*, i64, i64, i64, i64, i64, i64 }
%struct.st_ZAllocator = type { %struct.st_ZAllocator*, %struct.st_ZAllocator*, %struct.st_ZRow**, %struct.st_MallocBlock*, %struct.st_MallocBlock**, i8*, %struct.st_KMutex*, i64, i64, i64, i64, i64, i64, i32, i32, i64 }
%struct.st_MallocBlock = type { %struct.st_MallocBlock**, %struct.st_MallocBlock*, %struct.st_MallocBlock*, %struct.st_ZAllocator* }
%struct.st_KMutex = type { i8* }
%struct.st_ZBlock = type { %union.ZHeader }
%union.ZHeader = type { i64 }
%struct.st_MaxMemoryCell = type { i64, %struct.st_MaxMemoryCell* }
%struct.st_PageInformation = type { i64, i64, i64, i64, i32 }
%ident_t = type { i32, i32, i32, i32, i8* }
%"class.std::__1::complex" = type { double, double }
%"struct.std::__1::pair" = type { i64, i64 }
%struct.st_ParallelThreadsSchedule = type { i64, i64, i32 }
%struct.st_ParallelThreadsEnvironment = type { i64, i32, %struct.st_KMutex*, %struct.st_KMutex*, %struct.st_KMutex*, %struct.st_KMutex*, i64, %struct.st_ParallelThread*, %struct.st_ParallelThread**, %struct.st_ParallelThread*, i32, i32 }
%struct.st_ParallelThread = type { i64, i32, i32, %struct.st_KMutex*, %struct.st_KMutex*, %struct.st_ParallelThreadsEnvironment*, %struct.st_ParallelThreadState*, %struct.st_KMutex*, i64, %struct.st_ParallelThread* }
%struct.st_ParallelThreadState = type { i64, i64, i64, void (i8*, %struct.st_ParallelThread*)*, i8* }
%"struct.std::__1::pair.0" = type { %"class.std::__1::complex", %"class.std::__1::complex" }
%struct.st_MDataArray = type { double, i64*, i64, i32, i32, i32, i64, i8*, i64 }
%struct.st_WolframLibraryData = type { void (i8*)*, i32 (i64, i64, i64*, %struct.st_MDataArray**)*, void (%struct.st_MDataArray*)*, i32 (%struct.st_MDataArray*, %struct.st_MDataArray**)*, i64 (%struct.st_MDataArray*)*, void (%struct.st_MDataArray*)*, void (%struct.st_MDataArray*)*, i32 (%struct.st_MDataArray*, i64*, i64)*, i32 (%struct.st_MDataArray*, i64*, double)*, i32 (%struct.st_MDataArray*, i64*, double, double)*, i32 (%struct.st_MDataArray*, %struct.st_MDataArray*, i64*, i64)*, i32 (%struct.st_MDataArray*, i64*, i64*)*, i32 (%struct.st_MDataArray*, i64*, double*)*, i32 (%struct.st_MDataArray*, i64*, %"class.std::__1::complex"*)*, i32 (%struct.st_MDataArray*, i64*, i64, %struct.st_MDataArray**)*, i64 (%struct.st_MDataArray*)*, i64* (%struct.st_MDataArray*)*, i64 (%struct.st_MDataArray*)*, i64 (%struct.st_MDataArray*)*, i64* (%struct.st_MDataArray*)*, double* (%struct.st_MDataArray*)*, %"class.std::__1::complex"* (%struct.st_MDataArray*)*, void (i8*)*, i64 ()*, %struct.MLINK_STRUCT* (%struct.st_WolframLibraryData*)*, i32 (%struct.MLINK_STRUCT*)*, i32 (%struct.st_WolframLibraryData*, i8*, i32, i64, i8*)*, %struct.st_WolframRuntimeData*, %struct.st_WolframCompileLibrary_Functions*, i64, i32 (i8*, void (%struct.st_MInputStream*, i8*, i8*)*, i32 (i8*, i8*)*, i8*, void (i8*)*)*, i32 (i8*)*, i32 (i8*, void (%struct.st_MOutputStream*, i8*, i8*, i32)*, i32 (i8*, i8*)*, i8*, void (i8*)*)*, i32 (i8*)*, %struct.st_WolframIOLibrary_Functions*, %struct.MLENV_STRUCT* (%struct.st_WolframLibraryData*)*, %struct.st_WolframSparseLibrary_Functions*, %struct.st_WolframImageLibrary_Functions*, i32 (i8*, void (%struct.st_WolframLibraryData*, i32, i64)*)*, i32 (i8*)*, i32 (i8*, i64)*, i32 (i8*, i32 (%struct.st_WolframLibraryData*, i64, %struct.st_MDataArray*)*)*, i32 (i8*)*, i32 (i64, i64, %union.MArgument*, i32*)*, i32 (i64)*, i32 (i8*, i8)*, i32 ()*, %struct.st_WolframRawArrayLibrary_Functions*, %struct.st_WolframNumericArrayLibrary_Functions* }
%struct.MLINK_STRUCT = type opaque
%struct.st_WolframRuntimeData = type opaque
%struct.st_WolframCompileLibrary_Functions = type { i64, %struct.M_TENSOR_INITIALIZATION_DATA_STRUCT* (%struct.st_WolframLibraryData*, i64)*, void (%struct.M_TENSOR_INITIALIZATION_DATA_STRUCT*)*, void (%struct.st_WolframLibraryData*, i32)*, %struct.st_MDataArray* ()*, i32 (%struct.st_MDataArray**, i32, i64, i64*)*, i32 (%struct.st_MDataArray**, %struct.st_MDataArray*)*, i32 (%struct.st_MDataArray**, %struct.st_MDataArray*)*, i32 (%struct.st_MDataArray*, %struct.st_MDataArray*, i64*)*, i32 (%struct.st_MDataArray**, %struct.st_MDataArray*, i64*, i64)*, i32 (%struct.st_MDataArray**, %struct.st_MDataArray*, i64, i32*, i8**)*, i32 (%struct.st_MDataArray**, %struct.st_MDataArray*, i64, i32*, i8**)*, i64 (i8*, i8*, i64, i64*, i32)* (i32, i32)*, i32 (i32, i32, %struct.st_MDataArray*, i32, %struct.st_MDataArray**)*, i32 (i32, i32, i32, i8*, i32, i8*)*, i64 (i8*, i8*, i8*, i64, i64*, i32)* (i32, i32, i32)*, i32 (i32, i32, %struct.st_MDataArray*, %struct.st_MDataArray*, i32, %struct.st_MDataArray**)*, i32 (i32, i32, i32, i8*, i32, i8*, i32, i8*)*, i32 (i32, double, i64, double*)*, i32 (i32, double, i64, %"class.std::__1::complex"*, i32*)*, i8* (%struct.st_WolframLibraryData*, i8*)*, i32 (%struct.st_WolframLibraryData*, i8*, i64, i64, i64, i32*, i8**, i32, i64, i8*)*, i8** (%struct.st_WolframLibraryData*, i64)*, i32 (%struct.st_WolframLibraryData*, i64, %union.MArgument*, i32*)* (i8*, i8*)*, i32 (%struct.st_WolframLibraryData*, i64, %union.MArgument*, i32*)* (i8*)*, i32 (i8*, i32, i32)*, %struct.st_MDataArray* (i8*, i32, i64)*, i32 (%struct.st_MDataArray*, %struct.st_MDataArray**)* }
%struct.M_TENSOR_INITIALIZATION_DATA_STRUCT = type { %struct.st_MDataArray**, i64, i32 }
%union.MArgument = type { i32* }
%struct.st_MInputStream = type { i8*, i8*, i8*, i32, i32, i32, i32, i8*, i8*, i8*, %struct.st_MInputStream*, i64 (%struct.st_MInputStream*, i8*, i64)*, i32 (%struct.st_MInputStream*, i64)*, i32 (%struct.st_MInputStream*)*, i64 (%struct.st_MInputStream*)*, i8* (%struct.st_MInputStream*)*, void (%struct.st_MInputStream*)*, i64 (%struct.st_MInputStream*)*, void (%struct.st_MInputStream*, i8*)*, i32 (%struct.st_MInputStream*)*, i32 (%struct.st_MInputStream*)*, void (%struct.st_MInputStream*)*, i32 (%struct.st_MInputStream*)* }
%struct.st_MOutputStream = type { i8*, i8*, i8*, i32, i32, i32, i32, i8*, i8*, i8*, %struct.st_MOutputStream*, i64 (%struct.st_MOutputStream*, i8*, i64)*, i32 (%struct.st_MOutputStream*)*, i64 (%struct.st_MOutputStream*)*, i8* (%struct.st_MOutputStream*)*, void (%struct.st_MOutputStream*)*, i32 (%struct.st_MOutputStream*)*, void (%struct.st_MOutputStream*, i8*)*, i32 (%struct.st_MOutputStream*)* }
%struct.st_WolframIOLibrary_Functions = type { i64 ()*, i64 (void (i64, i8*)*, i8*)*, void (i64, i8*, %struct.st_DataStore*)*, i32 (i64)*, i32 (i64)*, %struct.st_DataStore* ()*, void (%struct.st_DataStore*, i64)*, void (%struct.st_DataStore*, double)*, void (%struct.st_DataStore*, double, double)*, void (%struct.st_DataStore*, i8*)*, void (%struct.st_DataStore*, %struct.st_MDataArray*)*, void (%struct.st_DataStore*, %struct.st_MDataArray*)*, void (%struct.st_DataStore*, %struct.IMAGEOBJ_ENTRY*)*, void (%struct.st_DataStore*, %struct.st_DataStore*)*, void (%struct.st_DataStore*, i8*, i64)*, void (%struct.st_DataStore*, i8*, double)*, void (%struct.st_DataStore*, i8*, double, double)*, void (%struct.st_DataStore*, i8*, i8*)*, void (%struct.st_DataStore*, i8*, %struct.st_MDataArray*)*, void (%struct.st_DataStore*, i8*, %struct.st_MDataArray*)*, void (%struct.st_DataStore*, i8*, %struct.IMAGEOBJ_ENTRY*)*, void (%struct.st_DataStore*, i8*, %struct.st_DataStore*)*, i64 (i64)*, void (%struct.st_DataStore*)*, %struct.st_DataStore* (%struct.st_DataStore*)*, i64 (%struct.st_DataStore*)*, %struct.DataStoreNode_t* (%struct.st_DataStore*)*, %struct.DataStoreNode_t* (%struct.st_DataStore*)*, %struct.DataStoreNode_t* (%struct.DataStoreNode_t*)*, i32 (%struct.DataStoreNode_t*)*, i32 (%struct.DataStoreNode_t*, %union.MArgument*)*, i32 (%struct.DataStoreNode_t*, i8**)*, void (%struct.st_DataStore*, i32)*, void (%struct.st_DataStore*, i8*, i32)*, void (%struct.st_DataStore*, %struct.st_MDataArray*)*, void (%struct.st_DataStore*, i8*, %struct.st_MDataArray*)*, void (%struct.st_DataStore*, %struct.st_MSparseArray*)*, void (%struct.st_DataStore*, i8*, %struct.st_MSparseArray*)* }
%struct.st_DataStore = type opaque
%struct.IMAGEOBJ_ENTRY = type { i8*, i64, i64, i64, i64, i32, i32, i32, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64*, i64, i64, i64, i64, i8*, double*, double*, i8*, i64, void (i8*, i8*, i64)*, void (i8*, i8*, i64, i64, i64)*, void (i8*, i8*, i64, i64, i64, i64, i64)*, void (i8*, i8*, i64, i64, i64, i64, i64, i64)*, void (%struct.IMAGEOBJ_ENTRY*, double, i8*)*, i8* (%struct.IMAGEOBJ_ENTRY*, double*, i64)*, i64, i64 }
%struct.DataStoreNode_t = type opaque
%struct.st_MSparseArray = type { %struct.st_SparseArrayAllocation*, %struct.st_MTensorSparseArrayAllocation*, i64*, i64*, %struct.st_TensorProperty*, i64, i32, i32 }
%struct.st_SparseArrayAllocation = type { %struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, i8* }
%struct.st_MTensorSparseArrayAllocation = type { %struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_SparseArraySkeleton* }
%struct.st_SparseArraySkeleton = type { i8*, i8*, i64*, i64*, %struct.st_TensorProperty*, i64, i32, i32 }
%struct.st_TensorProperty = type { double, i64*, i64, i32, i32 }
%struct.MLENV_STRUCT = type opaque
%struct.st_WolframSparseLibrary_Functions = type { i32 (%struct.st_MSparseArray*, %struct.st_MSparseArray**)*, void (%struct.st_MSparseArray*)*, void (%struct.st_MSparseArray*)*, void (%struct.st_MSparseArray*)*, i64 (%struct.st_MSparseArray*)*, i64 (%struct.st_MSparseArray*)*, i64* (%struct.st_MSparseArray*)*, %struct.st_MDataArray** (%struct.st_MSparseArray*)*, %struct.st_MDataArray** (%struct.st_MSparseArray*)*, %struct.st_MDataArray** (%struct.st_MSparseArray*)*, %struct.st_MDataArray** (%struct.st_MSparseArray*)*, i32 (%struct.st_MSparseArray*, %struct.st_MDataArray**)*, i32 (%struct.st_MSparseArray*, %struct.st_MDataArray*, %struct.st_MSparseArray**)*, i32 (%struct.st_MSparseArray*, %struct.st_MDataArray**)*, i32 (%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MSparseArray**)*, i32 (%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MSparseArray**)* }
%struct.st_WolframImageLibrary_Functions = type { i32 (i64, i64, i64, i32, i32, i32, %struct.IMAGEOBJ_ENTRY**)*, i32 (i64, i64, i64, i64, i32, i32, i32, %struct.IMAGEOBJ_ENTRY**)*, i32 (%struct.IMAGEOBJ_ENTRY*, %struct.IMAGEOBJ_ENTRY**)*, void (%struct.IMAGEOBJ_ENTRY*)*, void (%struct.IMAGEOBJ_ENTRY*)*, void (%struct.IMAGEOBJ_ENTRY*)*, i64 (%struct.IMAGEOBJ_ENTRY*)*, i32 (%struct.IMAGEOBJ_ENTRY*)*, i64 (%struct.IMAGEOBJ_ENTRY*)*, i64 (%struct.IMAGEOBJ_ENTRY*)*, i64 (%struct.IMAGEOBJ_ENTRY*)*, i64 (%struct.IMAGEOBJ_ENTRY*)*, i64 (%struct.IMAGEOBJ_ENTRY*)*, i32 (%struct.IMAGEOBJ_ENTRY*)*, i32 (%struct.IMAGEOBJ_ENTRY*)*, i32 (%struct.IMAGEOBJ_ENTRY*)*, i64 (%struct.IMAGEOBJ_ENTRY*)*, i32 (%struct.IMAGEOBJ_ENTRY*, i64*, i64, i8*)*, i32 (%struct.IMAGEOBJ_ENTRY*, i64*, i64, i8*)*, i32 (%struct.IMAGEOBJ_ENTRY*, i64*, i64, i16*)*, i32 (%struct.IMAGEOBJ_ENTRY*, i64*, i64, float*)*, i32 (%struct.IMAGEOBJ_ENTRY*, i64*, i64, double*)*, i32 (%struct.IMAGEOBJ_ENTRY*, i64*, i64, i8)*, i32 (%struct.IMAGEOBJ_ENTRY*, i64*, i64, i8)*, i32 (%struct.IMAGEOBJ_ENTRY*, i64*, i64, i16)*, i32 (%struct.IMAGEOBJ_ENTRY*, i64*, i64, float)*, i32 (%struct.IMAGEOBJ_ENTRY*, i64*, i64, double)*, i8* (%struct.IMAGEOBJ_ENTRY*)*, i8* (%struct.IMAGEOBJ_ENTRY*)*, i8* (%struct.IMAGEOBJ_ENTRY*)*, i16* (%struct.IMAGEOBJ_ENTRY*)*, float* (%struct.IMAGEOBJ_ENTRY*)*, double* (%struct.IMAGEOBJ_ENTRY*)*, %struct.IMAGEOBJ_ENTRY* (%struct.IMAGEOBJ_ENTRY*, i32, i32)* }
%struct.st_WolframRawArrayLibrary_Functions = type { i32 (i32, i64, i64*, %struct.st_MDataArray**)*, void (%struct.st_MDataArray*)*, i32 (%struct.st_MDataArray*, %struct.st_MDataArray**)*, void (%struct.st_MDataArray*)*, void (%struct.st_MDataArray*)*, i64 (%struct.st_MDataArray*)*, i32 (%struct.st_MDataArray*)*, i64 (%struct.st_MDataArray*)*, i64* (%struct.st_MDataArray*)*, i64 (%struct.st_MDataArray*)*, i8* (%struct.st_MDataArray*)*, %struct.st_MDataArray* (%struct.st_MDataArray*, i32)* }
%struct.st_WolframNumericArrayLibrary_Functions = type { i32 (i32, i64, i64*, %struct.st_MDataArray**)*, void (%struct.st_MDataArray*)*, i32 (%struct.st_MDataArray*, %struct.st_MDataArray**)*, void (%struct.st_MDataArray*)*, void (%struct.st_MDataArray*)*, i64 (%struct.st_MDataArray*)*, i32 (%struct.st_MDataArray*)*, i64 (%struct.st_MDataArray*)*, i64* (%struct.st_MDataArray*)*, i64 (%struct.st_MDataArray*)*, i8* (%struct.st_MDataArray*)*, i32 (%struct.st_MDataArray**, %struct.st_MDataArray*, i32, i32, double)* }
%struct.MemoryLinkedList_struct = type { i8*, %struct.MemoryLinkedList_struct* }
%struct.MBag_struct = type { %struct.MemoryLinkedList_struct*, %struct.MemoryLinkedList_struct**, i64, i64, i8*, i8*, i64, i64, i32 }
%struct.part_ind = type { i32, i64, i64*, i64, i64 }
%struct.MBagArray_struct = type { i64, i64, %struct.MBag_struct** }
%struct.RuntimeData_struct = type { i8*, %struct.MBagArray_struct*, %struct.BPdata*, %struct.MemoryLinkedList_struct*, i8*, i8*, i8*, %struct.MemoryLinkedList_struct*, %struct.MemoryLinkedList_struct*, i32, %struct.st_WolframLibraryData* }
%struct.BPdata = type opaque
%struct.RandomGenerator_struct = type { %struct.RandomGeneratorMethodData_struct*, %struct.RandomGeneratorFunctions_struct*, void (%struct.RandomGenerator_struct*, i64*, i64)*, i8* }
%struct.RandomGeneratorMethodData_struct = type { i64, i8*, i64 }
%struct.RandomGeneratorFunctions_struct = type { i64 (i64, %struct.RandomGenerator_struct*)*, i32 (i64*, i64, %struct.RandomGenerator_struct*)*, i32 (i64*, i64, i64, i64, %struct.RandomGenerator_struct*)*, i32 (double*, i64, double, double, %struct.RandomGenerator_struct*)*, i32 (i8**, i64, i8*, i8*, %struct.RandomGenerator_struct*)*, i32 (i8**, i64, i8*, i8*, double, %struct.RandomGenerator_struct*)*, %struct.RandomDistributionFunctionArray_struct* }
%struct.RandomDistributionFunctionArray_struct = type opaque
%struct.CompilerError_st = type { i32, i32 }
%struct._Unwind_Exception = type { i64, void (i32, %struct._Unwind_Exception*)*, i64, i64 }
%class.WolframRuntimeException = type <{ %"class.std::exception", i32, [4 x i8] }>
%"class.std::exception" = type { i32 (...)** }
%"class.std::__1::basic_string" = type { %"class.std::__1::__compressed_pair" }
%"class.std::__1::__compressed_pair" = type { %"struct.std::__1::__compressed_pair_elem" }
%"struct.std::__1::__compressed_pair_elem" = type { %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__rep" }
%"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__rep" = type { %union.anon }
%union.anon = type { %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__long" }
%"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__long" = type { i64, i64, i8* }
%"class.std::__1::complex.156" = type { float, float }
%"class.std::__1::basic_ostream" = type { i32 (...)**, %"class.std::__1::basic_ios.base" }
%"class.std::__1::basic_ios.base" = type <{ %"class.std::__1::ios_base", %"class.std::__1::basic_ostream"*, i32 }>
%"class.std::__1::ios_base" = type { i32 (...)**, i32, i64, i64, i32, i32, i8*, i8*, void (i32, %"class.std::__1::ios_base"*, i32)**, i32*, i64, i64, i64*, i64, i64, i8**, i64, i64 }
%"class.std::__1::basic_ostream<char, std::__1::char_traits<char> >::sentry" = type { i8, %"class.std::__1::basic_ostream"* }
%"class.std::__1::locale" = type { %"class.std::__1::locale::__imp"* }
%"class.std::__1::locale::__imp" = type opaque
%"class.std::__1::locale::facet" = type { %"class.std::__1::__shared_count" }
%"class.std::__1::__shared_count" = type { i32 (...)**, i64 }
%"class.std::__1::locale::id" = type <{ %union.ZHeader, i32, [4 x i8] }>
%"class.std::__1::basic_streambuf" = type { i32 (...)**, %"class.std::__1::locale", i8*, i8*, i8*, i8*, i8*, i8* }
%"class.std::__1::vector" = type { %"class.std::__1::__vector_base" }
%"class.std::__1::__vector_base" = type { %"class.std::__1::future"*, %"class.std::__1::future"*, %"class.std::__1::__compressed_pair.180" }
%"class.std::__1::future" = type { %"class.std::__1::__assoc_sub_state"* }
%"class.std::__1::__assoc_sub_state" = type <{ %"class.std::__1::__shared_count", %struct.st_KMutex, %"class.std::__1::mutex", %"class.std::__1::condition_variable", i32, [4 x i8] }>
%"class.std::__1::mutex" = type { %struct._opaque_pthread_mutex_t }
%struct._opaque_pthread_mutex_t = type { i64, [56 x i8] }
%"class.std::__1::condition_variable" = type { %struct._opaque_pthread_cond_t }
%struct._opaque_pthread_cond_t = type { i64, [40 x i8] }
%"class.std::__1::__compressed_pair.180" = type { %"struct.std::__1::__compressed_pair_elem.179" }
%"struct.std::__1::__compressed_pair_elem.179" = type { %"class.std::__1::future"* }
%"class.std::__1::__thread_struct" = type { %"class.std::__1::__thread_struct_imp"* }
%"class.std::__1::__thread_struct_imp" = type opaque
%struct._opaque_pthread_t = type { i64, %struct.__darwin_pthread_handler_rec*, [8176 x i8] }
%struct.__darwin_pthread_handler_rec = type { void (i8*)*, i8*, %struct.__darwin_pthread_handler_rec* }
%"class.std::__1::thread" = type { %struct._opaque_pthread_t* }
%union.anon.0 = type { i8 }
%"class.std::logic_error" = type { %"class.std::exception", %struct.st_KMutex }
%"class.std::length_error" = type { %"class.std::logic_error" }
%"class.std::__1::__async_assoc_state" = type { %"class.std::__1::__assoc_sub_state.base", %"class.std::__1::__async_func" }
%"class.std::__1::__assoc_sub_state.base" = type <{ %"class.std::__1::__shared_count", %struct.st_KMutex, %"class.std::__1::mutex", %"class.std::__1::condition_variable", i32 }>
%"class.std::__1::__async_func" = type { %"class.std::__1::tuple" }
%"class.std::__1::tuple" = type { %"struct.std::__1::__tuple_impl" }
%"struct.std::__1::__tuple_impl" = type { %"class.std::__1::__tuple_leaf" }
%"class.std::__1::__tuple_leaf" = type { %class.anon.8 }
%class.anon.8 = type { i64*, i64*, i64, i64, i64, %class.anon*, %"class.std::__1::vector.1"* }
%class.anon = type { i64 (i64)* }
%"class.std::__1::vector.1" = type { %"class.std::__1::__vector_base.2" }
%"class.std::__1::__vector_base.2" = type { i32*, i32*, %"class.std::__1::__compressed_pair.3" }
%"class.std::__1::__compressed_pair.3" = type { %union.MArgument }
%struct.st_IntegerArrayHashTable = type { %struct.st_hashtable*, i64, i8*, i32 }
%struct.st_hashtable = type { %struct.st_hash_entry**, i64 (%struct.st_hash_entry*)*, void (%struct.st_hash_entry*, %struct.st_hash_entry*)*, i32 (%struct.st_hash_entry*, %struct.st_hash_entry*)*, void (%struct.st_hash_entry*)*, void (%struct.st_hash_entry*)*, void (%struct.st_hashtable*)*, i64, i64, i64, i64, i64, i32 }
%struct.st_hash_entry = type { i8*, i8*, i64, %struct.st_hash_entry* }
%struct.st_WolframLibraryData.208 = type { void (i8*)*, i32 (i64, i64, i64*, %struct.st_MDataArray**)*, void (%struct.st_MDataArray*)*, i32 (%struct.st_MDataArray*, %struct.st_MDataArray**)*, i64 (%struct.st_MDataArray*)*, void (%struct.st_MDataArray*)*, void (%struct.st_MDataArray*)*, i32 (%struct.st_MDataArray*, i64*, i64)*, i32 (%struct.st_MDataArray*, i64*, double)*, i32 (%struct.st_MDataArray*, i64*, double, double)*, i32 (%struct.st_MDataArray*, %struct.st_MDataArray*, i64*, i64)*, i32 (%struct.st_MDataArray*, i64*, i64*)*, i32 (%struct.st_MDataArray*, i64*, double*)*, i32 (%struct.st_MDataArray*, i64*, %"class.std::__1::complex"*)*, i32 (%struct.st_MDataArray*, i64*, i64, %struct.st_MDataArray**)*, i64 (%struct.st_MDataArray*)*, i64* (%struct.st_MDataArray*)*, i64 (%struct.st_MDataArray*)*, i64 (%struct.st_MDataArray*)*, i64* (%struct.st_MDataArray*)*, double* (%struct.st_MDataArray*)*, %"class.std::__1::complex"* (%struct.st_MDataArray*)*, void (i8*)*, i64 ()*, %struct.MLINK_STRUCT* (%struct.st_WolframLibraryData.208*)*, i32 (%struct.MLINK_STRUCT*)*, i32 (%struct.st_WolframLibraryData.208*, i8*, i32, i64, i8*)*, %struct.st_WolframRuntimeData*, %struct.st_WolframCompileLibrary_Functions.194*, i64, i32 (i8*, void (%struct.st_MInputStream*, i8*, i8*)*, i32 (i8*, i8*)*, i8*, void (i8*)*)*, i32 (i8*)*, i32 (i8*, void (%struct.st_MOutputStream*, i8*, i8*, i32)*, i32 (i8*, i8*)*, i8*, void (i8*)*)*, i32 (i8*)*, %struct.st_WolframIOLibrary_Functions*, %struct.MLENV_STRUCT* (%struct.st_WolframLibraryData.208*)*, %struct.st_WolframSparseLibrary_Functions*, %struct.st_WolframImageLibrary_Functions*, i32 (i8*, void (%struct.st_WolframLibraryData.208*, i32, i64)*)*, i32 (i8*)*, i32 (i8*, i64)*, i32 (i8*, i32 (%struct.st_WolframLibraryData.208*, i64, %struct.st_MDataArray*)*)*, i32 (i8*)*, i32 (i64, i64, %union.MArgument*, i32*)*, i32 (i64)*, i32 (i8*, i8)*, i32 ()*, %struct.st_WolframRawArrayLibrary_Functions*, %struct.st_WolframNumericArrayLibrary_Functions* }
%struct.st_WolframCompileLibrary_Functions.194 = type { i64, %struct.M_TENSOR_INITIALIZATION_DATA_STRUCT* (%struct.st_WolframLibraryData.208*, i64)*, void (%struct.M_TENSOR_INITIALIZATION_DATA_STRUCT*)*, void (%struct.st_WolframLibraryData.208*, i32)*, %struct.st_MDataArray* ()*, i32 (%struct.st_MDataArray**, i32, i64, i64*)*, i32 (%struct.st_MDataArray**, %struct.st_MDataArray*)*, i32 (%struct.st_MDataArray**, %struct.st_MDataArray*)*, i32 (%struct.st_MDataArray*, %struct.st_MDataArray*, i64*)*, i32 (%struct.st_MDataArray**, %struct.st_MDataArray*, i64*, i64)*, i32 (%struct.st_MDataArray**, %struct.st_MDataArray*, i64, i32*, i8**)*, i32 (%struct.st_MDataArray**, %struct.st_MDataArray*, i64, i32*, i8**)*, i64 (i8*, i8*, i64, i64*, i32)* (i32, i32)*, i32 (i32, i32, %struct.st_MDataArray*, i32, %struct.st_MDataArray**)*, i32 (i32, i32, i32, i8*, i32, i8*)*, i64 (i8*, i8*, i8*, i64, i64*, i32)* (i32, i32, i32)*, i32 (i32, i32, %struct.st_MDataArray*, %struct.st_MDataArray*, i32, %struct.st_MDataArray**)*, i32 (i32, i32, i32, i8*, i32, i8*, i32, i8*)*, i32 (i32, double, i64, double*)*, i32 (i32, double, i64, %"class.std::__1::complex"*, i32*)*, i8* (%struct.st_WolframLibraryData.208*, i8*)*, i32 (%struct.st_WolframLibraryData.208*, i8*, i64, i64, i64, i32*, i8**, i32, i64, i8*)*, i8** (%struct.st_WolframLibraryData.208*, i64)*, {}* (i8*, i8*)*, {}* (i8*)*, i32 (i8*, i32, i32)*, %struct.st_MDataArray* (i8*, i32, i64)*, i32 (%struct.st_MDataArray*, %struct.st_MDataArray**)* }
%struct.MULTIPLE_ARRAY_HASH_STRUCT = type { i64, i64*, i64**, i64 }
%struct.sparse_matrix = type opaque
%struct.idc_num_t = type { i32, %"union.idc_num_t::num_t" }
%"union.idc_num_t::num_t" = type { %"class.std::__1::complex" }
%struct.part_message_data_struct = type { [2 x i64], void (i8*, %struct.part_message_data_struct*)*, i8*, i8* }
%"struct.std::__1::__hash_node_base" = type { %"struct.std::__1::__hash_node_base"* }
%"class.std::__1::__hash_table" = type <{ %"class.std::__1::unique_ptr.282", %"class.std::__1::__compressed_pair.10.284", %struct.st_ZBlock, %"class.std::__1::__compressed_pair.17", [4 x i8] }>
%"class.std::__1::unique_ptr.282" = type { %"class.std::__1::__compressed_pair.2" }
%"class.std::__1::__compressed_pair.2" = type { %"struct.std::__1::__compressed_pair_elem.3", %"struct.std::__1::__compressed_pair_elem.4" }
%"struct.std::__1::__compressed_pair_elem.3" = type { %"struct.std::__1::__hash_node_base"** }
%"struct.std::__1::__compressed_pair_elem.4" = type { %"class.std::__1::__bucket_list_deallocator" }
%"class.std::__1::__bucket_list_deallocator" = type { %struct.st_ZBlock }
%"class.std::__1::__compressed_pair.10.284" = type { %"struct.std::__1::__compressed_pair_elem.11.283" }
%"struct.std::__1::__compressed_pair_elem.11.283" = type { %"struct.std::__1::__hash_node_base" }
%"class.std::__1::__compressed_pair.17" = type { %"struct.half_float::detail::expr" }
%"struct.half_float::detail::expr" = type { float }
%"struct.std::__1::__hash_node_base.22" = type { %"struct.std::__1::__hash_node_base.22"* }
%"class.std::__1::__hash_table.26" = type <{ %"class.std::__1::unique_ptr.27", %"class.std::__1::__compressed_pair.36", %struct.st_ZBlock, %"class.std::__1::__compressed_pair.17", [4 x i8] }>
%"class.std::__1::unique_ptr.27" = type { %"class.std::__1::__compressed_pair.28" }
%"class.std::__1::__compressed_pair.28" = type { %"struct.std::__1::__compressed_pair_elem.29", %"struct.std::__1::__compressed_pair_elem.4" }
%"struct.std::__1::__compressed_pair_elem.29" = type { %"struct.std::__1::__hash_node_base.22"** }
%"class.std::__1::__compressed_pair.36" = type { %"struct.std::__1::__compressed_pair_elem.37" }
%"struct.std::__1::__compressed_pair_elem.37" = type { %"struct.std::__1::__hash_node_base.22" }
%struct.st_MNumericArrayConvertData = type { double, i32, i8 }
%struct.sched_param = type { i32, [4 x i8] }
%struct._opaque_pthread_mutexattr_t = type { i64, [8 x i8] }
%struct.timeval = type { i64, i32 }
%struct.RandomGeneratorMethod_struct = type { i8*, i64, %struct.RandomGenerator_struct* (i8*)*, %struct.RandomGenerator_struct* (%struct.RandomGenerator_struct*)*, void (%struct.RandomGenerator_struct*)*, %struct.RandomGeneratorMethod_struct*, i8* (%struct.RandomGenerator_struct*)*, %struct.RandomGenerator_struct* (i8*)*, i8* (i8*)*, i8* }
%struct.random_state_entry_struct = type { i32, %struct.RandomGenerator_struct*, %struct.random_state_entry_struct*, %struct.random_state_entry_struct* }
%struct.RandomState_struct = type { %struct.random_state_entry_struct*, i8*, i32 }
%struct.buffer_data_struct = type { i64*, i64, i64, i64, i64 }
%struct.RandomBuffer_struct = type { %struct.RandomGeneratorMethodData_struct*, %struct.RandomGeneratorFunctions_struct*, void (%struct.RandomGenerator_struct*, i64*, i64)*, %struct.RandomBufferState_struct* }
%struct.RandomBufferState_struct = type { %struct.buffer_data_struct*, i32 (i64*, i64*, i64*, %struct.RandomGenerator_struct*)*, %struct.RandomGenerator_struct* }
%struct.PositionTree_struct = type { i64, %struct.PositionTree_struct*, %struct.PositionTree_struct*, double }
%struct._VSLBRngProperties = type { i32, i32, i32, i32, i32, i32 (i32, i8*, i32, i32*)*, i32 (i8*, i32, float*, float, float)*, i32 (i8*, i32, double*, double, double)*, i32 (i8*, i32, i32*)* }
%struct.PMT32State_struct = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32* }

declare void @_Z14InitAllocationi(i32)

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

declare void @MCinterrupt()

declare i8* @kCalloc(i64, i64, i32, i32)

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1) #1

declare i8* @kRealloc(i8*, i64, i32, i32)

declare void @nomem()

declare i32 @puts(i8*)

declare void @exit(i32)

declare i8* @Malloc_fun(i64)

declare i8* @MallocAlign16_fun(i64)

declare i8* @MallocAlign_mreal_fun(i64)

declare i8* @Calloc_fun(i64, i64)

declare i8* @CallocAlign16_fun(i64, i64)

declare i8* @CallocAlign_mreal_fun(i64, i64)

declare i8* @Realloc_fun(i8*, i64)

declare i8* @ReallocAlign_mreal_fun(i8*, i64)

declare void @Free_fun(i8*)

declare i8* @Malloc_fun_for_gmp(i64)

declare i8* @Realloc_fun_for_gmp(i8*, i64, i64)

declare void @Free_fun_for_gmp(i8*, i64)

declare void @_Z10init_alloci(i32)

declare %struct.st_ZSpan**** @_Z26New_AllocationPageMapTablev()

declare i8* @malloc(i64)

declare void @_Z29Delete_AllocationPageMapTablePPPP8st_ZSpan(%struct.st_ZSpan****)

declare void @free(i8*)

declare %struct.st_ZSpan* @_Z29AllocationPageMap_addOrRemoveP9st_ZBlockmP8st_ZSpan(%struct.st_ZBlock*, i64, %struct.st_ZSpan*)

declare i64 @MemoryInUse()

declare i64 @MaxMemoryUsed()

declare i64 @MaxMemoryListPush(%struct.st_MaxMemoryCell**)

declare i64 @MaxMemoryListPop(%struct.st_MaxMemoryCell**)

declare i64 @MaxMemoryListLength(%struct.st_MaxMemoryCell*)

declare i64* @FullMemoryInformation(i64*, i64)

declare void @CompactifyMemory()

declare void @freeMemorySizeArray(i64*)

declare i64 @MCinuse()

declare i64 @MCliminuse()

declare void @MCsetlimit(i64)

declare void @MC_check(i64)

declare void @RecordCUDDMemory(i64)

declare i8* @_Z12ZPAGE_MALLOCm(i64)

declare i8* @"\01_mmap"(i8*, i64, i32, i32, i32, i64)

declare void @_Z10ZPAGE_FREEPvm(i8*, i64)

declare i32 @"\01_munmap"(i8*, i64)

declare %struct.st_PageInformation* @_Z19New_PageInformationv()

declare i64 @sysconf(i32)

; Function Attrs: nounwind readnone speculatable
declare i64 @llvm.ctlz.i64(i64, i1) #0

declare void @_Z22Delete_PageInformationP18st_PageInformation(%struct.st_PageInformation*)

declare i64 @getSystemMemory()

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

declare i32 @sysctl(i32*, i32, i8*, i64*, i8*, i64)

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

declare %struct.st_ZSpan* @_Z9New_ZSpanP7st_ZRow(%struct.st_ZRow*)

declare void @_Z12Delete_ZSpanP8st_ZSpan(%struct.st_ZSpan*)

declare void @_Z19MergeThreadSpanDataP7st_ZRow(%struct.st_ZRow*)

declare %struct.st_ZAllocator* @_Z19ZAllocator_getOrAddv()

declare %struct.st_ZAllocator* @_ZL14ZAllocator_addv()

declare void @_Z17ZAllocator_removev()

declare void @_Z17Delete_ZAllocatorP13st_ZAllocatori(%struct.st_ZAllocator*, i32)

declare void @_Z20FreeThreadMallocDataP13st_ZAllocator(%struct.st_ZAllocator*)

declare void @Delete_ZAllocatorList()

declare i8* @GetAllocatorList()

declare void @InitializeMemory()

declare void @ReclaimExitMemory()

declare i64 @zalloc_round(i64)

declare i8* @_Z11zalloc_nullmi(i64, i32)

declare i8* @zalloc(i64, i32, i32)

declare void @zfree(i8*)

declare i64 @zalloc_size(i8*)

declare i8* @zrealloc(i8*, i64, i32, i32)

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1) #1

declare void @_Z17InitRTLArithmetici(i32)

declare i32 @BigitBitLength(i64)

declare i32 @MintBitLength(i64)

declare i32 @BigitBitCount(i64)

; Function Attrs: nounwind readnone speculatable
declare i64 @llvm.ctpop.i64(i64) #0

declare i32 @MintBitCount(i64)

declare i32 @numzbitsbigit(i64)

; Function Attrs: nounwind readnone speculatable
declare i64 @llvm.cttz.i64(i64, i1) #0

declare i32 @numzbitsint(i64)

declare i64* @New_BV(i64)

declare void @BV_Mark(i64*, i64)

declare i64 @BV_SetQ(i64*, i64)

declare i32 @CompareDoubleDouble(double, double, double)

declare i32 @__gxx_personality_v0(...)

declare double @modf(double, double*)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.exp2.f64(double) #0

declare double @ldexp(double, i32)

declare i32 @CompareFloatFloat(float, float, float)

declare float @modff(float, float*)

; Function Attrs: nounwind readnone speculatable
declare float @llvm.exp2.f32(float) #0

declare float @ldexpf(float, i32)

declare i32 @near1_double(double, double, double)

declare i32 @near1_float(float, float, float)

declare i32 @near1_reim_mcomplex(double, double, double)

declare double @ToleranceShiftFactor(double)

declare float @ToleranceShiftFactorFloat(float)

declare i32 @TestException(double)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double) #0

declare i64 @machfunc_i_total(i8*, i8*, i64, i64*, i32)

declare i64 @_ZN10vfun_totalIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_totalIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN3wrt9scalar_op6binaryL7plus_opIxxLNS_13runtime_flagsE1EEENSt3__19enable_ifIXaasr11is_integralIT_EE5valuesr11is_integralIT0_EE5valueENS1_6detail7op_infoILNS8_2opE19EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE(i64*, i64*)

declare i64 @_ZN10vfun_totalIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_totalIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN3wrt9scalar_op6binaryL7plus_opIxxLNS_13runtime_flagsE0EEENSt3__19enable_ifIXaasr11is_integralIT_EE5valuesr11is_integralIT0_EE5valueENS1_6detail7op_infoILNS8_2opE19EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE(i64*, i64*)

declare i64 @_ZL11vfun1_0_runIxxExPFxPT_PKT0_RKxPS5_EPFS0_RKS0_SB_ES1_S4_S6_S7_(i64 (i64*, i64*, i64*, i64*)*, i64 (i64*, i64*)*, i64*, i64*, i64*, i64*)

declare void @.omp_outlined.(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (i64*, i64*, i64*, i64*)**, i64**, i64**, i64**, i64 (i64*, i64*)**)

declare void @__kmpc_fork_call(%ident_t*, i32, void (i32*, i32*, ...)*, ...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #2

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()

declare i32 @__kmpc_global_thread_num(%ident_t*)

declare void @__kmpc_for_static_init_8(%ident_t*, i32, i32, i32*, i64*, i64*, i64*, i64, i64)

declare void @__kmpc_critical(%ident_t*, i32, [8 x i32]*)

declare void @__kmpc_end_critical(%ident_t*, i32, [8 x i32]*)

declare i32 @omp_get_thread_num()

declare void @__clang_call_terminate(i8*)

declare void @__kmpc_for_static_fini(%ident_t*, i32)

declare void @_ZSt9terminatev()

; Function Attrs: nounwind readnone speculatable
declare { i64, i1 } @llvm.sadd.with.overflow.i64(i64, i64) #0

declare i8* @__cxa_allocate_exception(i64)

declare void @__cxa_throw(i8*, i8*, i8*)

declare i64 @machfunc_d_total(i8*, i8*, i64, i64*, i32)

declare i64 @_ZN10vfun_totalIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_totalIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare double @_ZN3wrt9scalar_op6binaryL7plus_opIddLNS_13runtime_flagsE1EEENSt3__19enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNS8_2opE19EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE(double*, double*)

declare i64 @_ZN10vfun_totalIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_totalIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare double @_ZN3wrt9scalar_op6binaryL7plus_opIddLNS_13runtime_flagsE0EEENSt3__19enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNS8_2opE19EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE(double*, double*)

declare i64 @_ZL11vfun1_0_runIddExPFxPT_PKT0_RKxPS5_EPFS0_RKS0_SB_ES1_S4_S6_S7_(i64 (double*, double*, i64*, i64*)*, double (double*, double*)*, double*, double*, i64*, i64*)

declare void @.omp_outlined..2(i32*, i32*, i64*, i64*, i64*, double*, i64*, i64*, i64*, i64*, i64 (double*, double*, i64*, i64*)**, double**, i64**, double**, double (double*, double*)**)

declare i32 @ippsSum_64f(double*, i32, double*)

declare i32 @__fpclassifyd(double)

declare i32 @feclearexcept(i32)

declare i32 @fetestexcept(i32)

declare i64 @machfunc_c_total(i8*, i8*, i64, i64*, i32)

declare i64 @_ZN10vfun_totalINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_totalINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL7plus_opINSt3__17complexIdEES5_LNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE19EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(%"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare i64 @_ZN10vfun_totalINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_totalINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL7plus_opINSt3__17complexIdEES5_LNS_13runtime_flagsE0EEENS3_9enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE19EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(%"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare void @.omp_outlined..4(i32*, i32*, i64*, i64*, i64*, %"class.std::__1::complex"*, i64*, i64*, i64*, i64*, i64 (%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)**, %"class.std::__1::complex"**, i64**, %"class.std::__1::complex"**, { double, double } (%"class.std::__1::complex"*, %"class.std::__1::complex"*)**)

declare i32 @ippsSum_64fc(%"class.std::__1::complex"*, i32, %"class.std::__1::complex"*)

declare i64 @machfunc_i_min(i8*, i8*, i64, i64*, i32)

declare i64 @_ZN8vfun_minIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_minIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN3wrt9scalar_op6binaryL6min_opIxxLNS_13runtime_flagsE1EEENSt3__19enable_ifIXaasr11is_integralIT_EE5valuesr11is_integralIT0_EE5valueENS1_6detail7op_infoILNS8_2opE16EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE(i64*, i64*)

declare i64 @_ZN8vfun_minIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_minIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN3wrt9scalar_op6binaryL6min_opIxxLNS_13runtime_flagsE0EEENSt3__19enable_ifIXaasr11is_integralIT_EE5valuesr11is_integralIT0_EE5valueENS1_6detail7op_infoILNS8_2opE16EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE(i64*, i64*)

declare i64 @machfunc_d_min(i8*, i8*, i64, i64*, i32)

declare i64 @_ZN8vfun_minIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_minIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare double @_ZN3wrt9scalar_op6binaryL6min_opIddLNS_13runtime_flagsE1EEENSt3__19enable_ifIXooaasr17is_floating_pointIT_EE5valuentsr10is_complexIT0_EE5valueaasr17is_floating_pointIS7_EE5valuentsr10is_complexIS6_EE5valueENS1_6detail7op_infoILNS8_2opE16EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE(double*, double*)

declare i64 @_ZN8vfun_minIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_minIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare double @_ZN3wrt9scalar_op6binaryL6min_opIddLNS_13runtime_flagsE0EEENSt3__19enable_ifIXooaasr17is_floating_pointIT_EE5valuentsr10is_complexIT0_EE5valueaasr17is_floating_pointIS7_EE5valuentsr10is_complexIS6_EE5valueENS1_6detail7op_infoILNS8_2opE16EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE(double*, double*)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.minnum.f64(double, double) #0

declare i32 @ippsMin_64f(double*, i32, double*)

declare i64 @machfunc_i_max(i8*, i8*, i64, i64*, i32)

declare i64 @_ZN8vfun_maxIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_maxIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN3wrt9scalar_op6binaryL6max_opIxxLNS_13runtime_flagsE1EEENSt3__19enable_ifIXaasr11is_integralIT_EE5valuesr11is_integralIT0_EE5valueENS1_6detail7op_infoILNS8_2opE15EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE(i64*, i64*)

declare i64 @_ZN8vfun_maxIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_maxIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN3wrt9scalar_op6binaryL6max_opIxxLNS_13runtime_flagsE0EEENSt3__19enable_ifIXaasr11is_integralIT_EE5valuesr11is_integralIT0_EE5valueENS1_6detail7op_infoILNS8_2opE15EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE(i64*, i64*)

declare i64 @machfunc_d_max(i8*, i8*, i64, i64*, i32)

declare i64 @_ZN8vfun_maxIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_maxIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare double @_ZN3wrt9scalar_op6binaryL6max_opIddLNS_13runtime_flagsE1EEENSt3__19enable_ifIXooaasr17is_floating_pointIT_EE5valuentsr10is_complexIT0_EE5valueaasr17is_floating_pointIS7_EE5valuentsr10is_complexIS6_EE5valueENS1_6detail7op_infoILNS8_2opE15EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE(double*, double*)

declare i64 @_ZN8vfun_maxIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_maxIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare double @_ZN3wrt9scalar_op6binaryL6max_opIddLNS_13runtime_flagsE0EEENSt3__19enable_ifIXooaasr17is_floating_pointIT_EE5valuentsr10is_complexIT0_EE5valueaasr17is_floating_pointIS7_EE5valuentsr10is_complexIS6_EE5valueENS1_6detail7op_infoILNS8_2opE15EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE(double*, double*)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.maxnum.f64(double, double) #0

declare i32 @ippsMax_64f(double*, i32, double*)

declare i64 @machfunc_i_minmax(i8*, i8*, i64, i64*, i32)

declare i64 @_ZN11vfun_minmaxINSt3__14pairIxxEExE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKxRS9_SA_(%"struct.std::__1::pair"*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_minmaxINSt3__14pairIxxEExE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKxRS9_SA_(%"struct.std::__1::pair"*, i64*, i64*, i64*)

declare { i64, i64 } @_ZN3wrt9scalar_op6binaryL9minmax_opINSt3__14pairIxxEES5_LNS_13runtime_flagsE1EEENS3_9enable_ifIXaasr7is_pairIT_EE5valuesr7is_pairIT0_EE5valueENS1_6detail7op_infoILNSA_2opE17EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNS4_INSI_19first_argument_typeESM_EERKNS4_INSI_20second_argument_typeESQ_EE(%"struct.std::__1::pair"*, %"struct.std::__1::pair"*)

declare i64 @_ZN11vfun_minmaxINSt3__14pairIxxEExE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKxRS9_SA_(%"struct.std::__1::pair"*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_minmaxINSt3__14pairIxxEExE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKxRS9_SA_(%"struct.std::__1::pair"*, i64*, i64*, i64*)

declare { i64, i64 } @_ZN3wrt9scalar_op6binaryL9minmax_opINSt3__14pairIxxEES5_LNS_13runtime_flagsE0EEENS3_9enable_ifIXaasr7is_pairIT_EE5valuesr7is_pairIT0_EE5valueENS1_6detail7op_infoILNSA_2opE17EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNS4_INSI_19first_argument_typeESM_EERKNS4_INSI_20second_argument_typeESQ_EE(%"struct.std::__1::pair"*, %"struct.std::__1::pair"*)

declare void @.omp_outlined..6(i32*, i32*, i64*, i64*, i64*, %"struct.std::__1::pair"*, i64*, i64*, i64*, i64*, i64 (%"struct.std::__1::pair"*, i64*, i64*, i64*)**, i64**, i64**, %"struct.std::__1::pair"**, { i64, i64 } (%"struct.std::__1::pair"*, %"struct.std::__1::pair"*)**)

declare i64 @machfunc_d_minmax(i8*, i8*, i64, i64*, i32)

declare i64 @_ZN11vfun_minmaxINSt3__14pairIddEEdE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKdRKxPSB_(%"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_minmaxINSt3__14pairIddEEdE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKdRKxPSB_(%"class.std::__1::complex"*, double*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL9minmax_opINSt3__14pairIddEES5_LNS_13runtime_flagsE1EEENS3_9enable_ifIXaasr7is_pairIT_EE5valuesr7is_pairIT0_EE5valueENS1_6detail7op_infoILNSA_2opE17EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNS4_INSI_19first_argument_typeESM_EERKNS4_INSI_20second_argument_typeESQ_EE(%"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare i64 @_ZN11vfun_minmaxINSt3__14pairIddEEdE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKdRKxPSB_(%"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_minmaxINSt3__14pairIddEEdE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKdRKxPSB_(%"class.std::__1::complex"*, double*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL9minmax_opINSt3__14pairIddEES5_LNS_13runtime_flagsE0EEENS3_9enable_ifIXaasr7is_pairIT_EE5valuesr7is_pairIT0_EE5valueENS1_6detail7op_infoILNSA_2opE17EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNS4_INSI_19first_argument_typeESM_EERKNS4_INSI_20second_argument_typeESQ_EE(%"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare void @.omp_outlined..8(i32*, i32*, i64*, i64*, i64*, %"class.std::__1::complex"*, i64*, i64*, i64*, i64*, i64 (%"class.std::__1::complex"*, double*, i64*, i64*)**, double**, i64**, %"class.std::__1::complex"**, { double, double } (%"class.std::__1::complex"*, %"class.std::__1::complex"*)**)

declare i32 @ippsMinMax_64f(double*, i32, double*, double*)

declare i64 @machfunc_i_maxabs(i8*, i8*, i64, i64*, i32)

declare i64 @_ZN11vfun_maxabsIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @machfunc_d_maxabs(i8*, i8*, i64, i64*, i32)

declare i64 @_ZN11vfun_maxabsIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i32 @ippsMaxAbs_64f(double*, i32, double*)

declare i64 @machfunc_c_maxabs(i8*, i8*, i64, i64*, i32)

declare i64 @_ZN11vfun_maxabsIdNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPdPKS2_RKxPSB_(double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIdNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPdPKS2_RKxPSB_(double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIdNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPdPKS2_RKxPSB_(double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIdNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPdPKS2_RKxPSB_(double*, %"class.std::__1::complex"*, i64*, i64*)

declare void @.omp_outlined..10(i32*, i32*, i64*, i64*, i64*, double*, i64*, i64*, i64*, i64*, i64 (double*, %"class.std::__1::complex"*, i64*, i64*)**, %"class.std::__1::complex"**, i64**, double**, double (double*, double*)**)

declare double @hypot(double, double)

declare i64 @machfunc_d_fpexception(i8*, i8*, i64, i64*, i32)

declare i64 @_ZN16vfun_fpexceptionIxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN16vfun_fpexceptionIxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN3wrt9scalar_op6binaryL9bit_or_opIxxLNS_13runtime_flagsE1EEENSt3__19enable_ifIXaasr18is_signed_integralIT_EE5valuesr18is_signed_integralIT0_EE5valueENS1_6detail7op_infoILNS8_2opE4EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE(i64*, i64*)

declare i64 @_ZN16vfun_fpexceptionIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN16vfun_fpexceptionIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN3wrt9scalar_op6binaryL9bit_or_opIxxLNS_13runtime_flagsE0EEENSt3__19enable_ifIXaasr18is_signed_integralIT_EE5valuesr18is_signed_integralIT0_EE5valueENS1_6detail7op_infoILNS8_2opE4EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE(i64*, i64*)

declare void @.omp_outlined..12(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (i64*, double*, i64*, i64*)**, double**, i64**, i64**, i64 (i64*, i64*)**)

declare i64 @machfunc_c_fpexception(i8*, i8*, i64, i64*, i32)

declare i64 @_ZN16vfun_fpexceptionIxNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPxPKS2_RKxPSB_(i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN16vfun_fpexceptionIxNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPxPKS2_RKxPSB_(i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN16vfun_fpexceptionIxNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPxPKS2_RKxPSB_(i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN16vfun_fpexceptionIxNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPxPKS2_RKxPSB_(i64*, %"class.std::__1::complex"*, i64*, i64*)

declare void @.omp_outlined..14(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (i64*, %"class.std::__1::complex"*, i64*, i64*)**, %"class.std::__1::complex"**, i64**, i64**, i64 (i64*, i64*)**)

declare i64 @machfunc_d_sin(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_sin(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_cos(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_cos(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_tan(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_tan(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL14machfunc_d_cscPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL14machfunc_c_cscPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL14machfunc_d_secPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL14machfunc_c_secPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL14machfunc_d_cotPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL14machfunc_c_cotPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_sinh(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_sinh(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_cosh(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_cosh(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_tanh(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_tanh(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_d_cschPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_c_cschPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_d_sechPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_c_sechPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_d_cothPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_c_cothPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_asin(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_asin(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_acos(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_acos(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_atan(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_atan(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_d_acscPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_c_acscPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_d_asecPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_c_asecPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_d_acotPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_c_acotPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_asinh(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_asinh(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_acosh(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_acosh(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_atanh(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_atanh(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL16machfunc_d_acschPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL16machfunc_c_acschPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL16machfunc_d_asechPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL16machfunc_c_asechPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL16machfunc_d_acothPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL16machfunc_c_acothPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_exp(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_exp(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_expm1(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_expm1(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_log(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_log(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_log1p(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_log1p(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_i_log2(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_log2(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_log2(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_i_log10(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_log10(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_log10(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_i_abs(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_abs(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_abs(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL14machfunc_i_argPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL14machfunc_d_argPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_arg(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_i_conj(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_conj(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_conj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL13machfunc_i_imPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL13machfunc_d_imPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL13machfunc_c_imPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL13machfunc_i_rePvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL13machfunc_d_rePvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL13machfunc_c_rePvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_i_minus(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_minus(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_minus(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_i_signPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_d_signPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_c_signPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL14machfunc_i_posPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL14machfunc_d_posPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL14machfunc_i_negPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL14machfunc_d_negPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_i_nposPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_d_nposPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_i_nnegPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_d_nnegPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL16machfunc_i_roundPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_round(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL16machfunc_i_floorPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_floor(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL18machfunc_i_ceilingPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_ceiling(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL19machfunc_i_fracpartPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_fracpart(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_fracpart(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL18machfunc_i_intpartPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_intpart(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL16machfunc_i_evenqPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL16machfunc_d_evenqPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_i_oddqPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_d_oddqPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_i_square(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_square(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_square(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_sqrt(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_sqrt(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_cbrt(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_rsqrt(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_rsqrt(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_recip(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_recip(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL17machfunc_i_bitnotPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL20machfunc_i_bitlengthPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_i_intexp(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_i_intlen(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_sinc(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_sinc(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_i_fibonacci(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_fibonacci(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_fibonacci(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_i_lucasl(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_lucasl(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_lucasl(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_gudermannian(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_gudermannian(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_inversegudermannian(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_inversegudermannian(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_haversine(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_haversine(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_inversehaversine(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_inversehaversine(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_erfc(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_erf(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_i_gamma(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_gamma(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_loggamma(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL18machfunc_i_unitizePvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL18machfunc_d_unitizePvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL18machfunc_c_unitizePvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_i_mod1PvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_d_mod1PvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_c_mod1PvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_logistic(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_logistic(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_i_rampPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_d_rampPvPKvxPKxj(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_i_abssquare(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_d_abssquare(i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_c_abssquare(i8*, i8*, i64, i64*, i32)

declare i64 @_ZN14vfun_abssquareIdNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPdPKS2_RKxPSB_(double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN14vfun_abssquareIdNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPdPKS2_RKxPSB_(double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN14vfun_abssquareIdNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPdPKS2_RKxPSB_(double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN14vfun_abssquareIdNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPdPKS2_RKxPSB_(double*, %"class.std::__1::complex"*, i64*, i64*)

declare void @.omp_outlined..8.23(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (double*, %"class.std::__1::complex"*, i64*, i64*)**, double**, %"class.std::__1::complex"**, i64**)

declare i64 @_ZN14vfun_abssquareIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN14vfun_abssquareIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN14vfun_abssquareIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN14vfun_abssquareIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN14vfun_abssquareIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN14vfun_abssquareIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare void @.omp_outlined..2.25(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (double*, double*, i64*, i64*)**, double**, double**, i64**)

declare i64 @_ZN14vfun_abssquareIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN14vfun_abssquareIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN14vfun_abssquareIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN14vfun_abssquareIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare void @.omp_outlined..6.26(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (i64*, i64*, i64*, i64*)**, i64**, i64**, i64**)

; Function Attrs: nounwind readnone speculatable
declare { i64, i1 } @llvm.smul.with.overflow.i64(i64, i64) #0

declare i64 @_ZN9vfun_rampIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_rampIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_rampIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_rampIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_rampIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_rampIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_rampIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_rampIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_rampIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_rampIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_logisticINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_logisticINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_logisticINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_logisticINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_logisticINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_logisticINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare void @.omp_outlined..4.27(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)**, %"class.std::__1::complex"**, %"class.std::__1::complex"**, i64**)

declare { double, double } @_ZN3wrt9scalar_op5unaryL11logistic_opINSt3__17complexIdEELNS_13runtime_flagsE0EEENS3_9enable_ifIXsr28is_floating_point_or_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE50EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE(%"class.std::__1::complex"*)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.exp.f64(double) #0

; Function Attrs: nounwind readnone speculatable
declare double @llvm.cos.f64(double) #0

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sin.f64(double) #0

declare { double, double } @_ZNSt3__1dvIdEENS_7complexIT_EERKS3_S5_(%"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare double @logb(double)

declare double @scalbn(double, i32)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.copysign.f64(double, double) #0

; Function Attrs: nounwind readnone speculatable
declare <2 x double> @llvm.fabs.v2f64(<2 x double>) #0

; Function Attrs: nounwind readnone speculatable
declare <2 x double> @llvm.copysign.v2f64(<2 x double>, <2 x double>) #0

declare { double, double } @_ZN3wrt9scalar_op5unaryL11logistic_opINSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXsr28is_floating_point_or_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE50EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE(%"class.std::__1::complex"*)

declare i64 @_ZN13vfun_logisticIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_logisticIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_logisticIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_logisticIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_logisticIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_logisticIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare double @_ZN3wrt9scalar_op5unaryL11logistic_opIdLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr28is_floating_point_or_complexIT_EE5valueENS1_6detail7op_infoILNS7_2opE50EN7rt_typeIS6_E4typeEE11result_typeEE4typeERKNSD_13argument_typeE(double)

declare i64 @_ZN9vfun_mod1INSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_mod1INSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_mod1INSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_mod1INSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_mod1INSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_mod1INSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_mod1IddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_mod1IddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_mod1IddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_mod1IddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_mod1IddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_mod1IddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_mod1IxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_mod1IxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_mod1IxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_mod1IxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPxPKS2_RKxPSB_(i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPxPKS2_RKxPSB_(i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPxPKS2_RKxPSB_(i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPxPKS2_RKxPSB_(i64*, %"class.std::__1::complex"*, i64*, i64*)

declare void @.omp_outlined..28(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (i64*, %"class.std::__1::complex"*, i64*, i64*)**, i64**, %"class.std::__1::complex"**, i64**)

declare i64 @_ZN12vfun_unitizeIxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare void @.omp_outlined..10.29(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (i64*, double*, i64*, i64*)**, i64**, double**, i64**)

declare i64 @_ZN12vfun_unitizeIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_loggammaIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_loggammaIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_loggammaIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_loggammaIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_loggammaIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_loggammaIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare double @lgamma(double)

declare i64 @_ZN10vfun_gammaIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_gammaIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_gammaIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_gammaIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_gammaIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_gammaIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare double @tgamma(double)

declare i64 @_ZN10vfun_gammaIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_gammaIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_gammaIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_gammaIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_erfIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_erfIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_erfIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_erfIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_erfIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_erfIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare double @erf(double)

declare i64 @_ZN9vfun_erfcIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_erfcIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_erfcIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_erfcIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_erfcIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_erfcIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare double @erfc(double)

declare i64 @_ZN21vfun_inversehaversineINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN21vfun_inversehaversineINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN21vfun_inversehaversineINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN21vfun_inversehaversineINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN21vfun_inversehaversineINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN21vfun_inversehaversineINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op5unaryL19inversehaversine_opINSt3__17complexIdEELNS_13runtime_flagsE0EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE44EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE(double, double)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sqrt.f64(double) #0

declare { double, double } @_ZN3wrt9scalar_op5unaryL8asinh_opINSt3__17complexIdEELNS_13runtime_flagsE0EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE13EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE(%"class.std::__1::complex"*)

declare double @frexp(double, i32*)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.log.f64(double) #0

declare double @atan2(double, double)

declare { double, double } @_ZN3wrt9scalar_op5unaryL8log1p_opINSt3__17complexIdEELNS_13runtime_flagsE0EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE47EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE(double, double)

declare double @log1p(double)

declare { double, double } @_ZN3wrt9scalar_op5unaryL19inversehaversine_opINSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE44EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE(double, double)

declare i64 @_ZN21vfun_inversehaversineIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN21vfun_inversehaversineIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN21vfun_inversehaversineIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN21vfun_inversehaversineIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN21vfun_inversehaversineIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN21vfun_inversehaversineIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare double @asin(double)

declare double @_ZN3wrt9scalar_op5unaryL19inversehaversine_opIdLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr17is_floating_pointIT_EE5valueENS1_6detail7op_infoILNS7_2opE44EN7rt_typeIS6_E4typeEE11result_typeEE4typeERKNSD_13argument_typeE(double)

declare i64 @_ZN14vfun_haversineINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN14vfun_haversineINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN14vfun_haversineINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN14vfun_haversineINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN14vfun_haversineINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN14vfun_haversineINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare double @sinh(double)

declare double @cosh(double)

declare { double, double } @_ZN3wrt9scalar_op5unaryL12haversine_opINSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXsr28is_floating_point_or_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE40EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE(%"class.std::__1::complex"*)

declare i64 @_ZN14vfun_haversineIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN14vfun_haversineIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN14vfun_haversineIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN14vfun_haversineIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN14vfun_haversineIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN14vfun_haversineIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN24vfun_inversegudermannianINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN24vfun_inversegudermannianINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN24vfun_inversegudermannianINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN24vfun_inversegudermannianINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN24vfun_inversegudermannianINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN24vfun_inversegudermannianINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op5unaryL22inversegudermannian_opINSt3__17complexIdEELNS_13runtime_flagsE0EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE43EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE(%"class.std::__1::complex"*)

declare { double, double } @_ZN3wrt9scalar_op5unaryL22inversegudermannian_opINSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE43EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE(%"class.std::__1::complex"*)

declare i64 @_ZN24vfun_inversegudermannianIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN24vfun_inversegudermannianIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN24vfun_inversegudermannianIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN24vfun_inversegudermannianIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN24vfun_inversegudermannianIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN24vfun_inversegudermannianIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare double @tan(double)

declare double @atanh(double)

declare double @_ZN3wrt9scalar_op5unaryL22inversegudermannian_opIdLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr17is_floating_pointIT_EE5valueENS1_6detail7op_infoILNS7_2opE43EN7rt_typeIS6_E4typeEE11result_typeEE4typeERKNSD_13argument_typeE(double)

declare i64 @_ZN17vfun_gudermannianINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN17vfun_gudermannianINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN17vfun_gudermannianINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN17vfun_gudermannianINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN17vfun_gudermannianINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN17vfun_gudermannianINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op5unaryL8atanh_opINSt3__17complexIdEELNS_13runtime_flagsE0EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE15EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE(%"class.std::__1::complex"*)

declare { double, double } @_ZN3wrt9scalar_op5unaryL15gudermannian_opINSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE39EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE(double, double)

declare i64 @_ZN17vfun_gudermannianIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN17vfun_gudermannianIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN17vfun_gudermannianIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN17vfun_gudermannianIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN17vfun_gudermannianIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN17vfun_gudermannianIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare double @tanh(double)

declare double @atan(double)

declare i64 @_ZN11vfun_lucaslINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_lucaslINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_lucaslINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_lucaslINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_lucaslINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_lucaslINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_lucaslIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_lucaslIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_lucaslIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_lucaslIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_lucaslIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_lucaslIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.pow.f64(double, double) #0

declare i64 @_ZN11vfun_lucaslIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_lucaslIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_lucaslIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_lucaslIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN14vfun_fibonacciINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN14vfun_fibonacciINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN14vfun_fibonacciINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN14vfun_fibonacciINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN14vfun_fibonacciINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN14vfun_fibonacciINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op5unaryL12fibonacci_opINSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXsr28is_floating_point_or_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE34EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE(%"class.std::__1::complex"*)

declare i64 @_ZN14vfun_fibonacciIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN14vfun_fibonacciIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN14vfun_fibonacciIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN14vfun_fibonacciIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN14vfun_fibonacciIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN14vfun_fibonacciIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN14vfun_fibonacciIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN14vfun_fibonacciIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN14vfun_fibonacciIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN14vfun_fibonacciIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_sincINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_sincINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_sincINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_sincINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_sincINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_sincINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL9divide_opINSt3__17complexIdEES5_LNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE9EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(%"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare i64 @_ZN9vfun_sincIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_sincIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_sincIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_sincIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_sincIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_sincIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN14vfun_bitlengthIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN14vfun_bitlengthIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN14vfun_bitlengthIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN14vfun_bitlengthIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_bitnotIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_bitnotIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_bitnotIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_bitnotIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_recipINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_recipINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_recipINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_recipINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_recipINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_recipINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_recipIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_recipIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_recipIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_recipIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_recipIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_recipIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i32 @vmlClearErrStatus()

declare void @vdInv(i64, double*, double*)

declare i32 @vmlGetErrStatus()

declare i64 @_ZN10vfun_rsqrtINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_rsqrtINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_rsqrtINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_rsqrtINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_rsqrtINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_rsqrtINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_rsqrtIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_rsqrtIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_rsqrtIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_rsqrtIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_rsqrtIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_rsqrtIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_cbrtIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_cbrtIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_cbrtIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_cbrtIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_cbrtIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_cbrtIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare double @cbrt(double)

declare i64 @_ZN9vfun_sqrtINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_sqrtINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_sqrtINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_sqrtINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_sqrtINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_sqrtINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare void @vzSqrt(i64, %"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare i64 @_ZN9vfun_sqrtIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_sqrtIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_sqrtIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_sqrtIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_sqrtIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_sqrtIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare void @vdSqrt(i64, double*, double*)

declare i64 @_ZN11vfun_squareINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_squareINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_squareINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_squareINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_squareINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_squareINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_squareIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_squareIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_squareIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_squareIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_squareIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_squareIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare void @vdSqr(i64, double*, double*)

declare i64 @_ZN11vfun_squareIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_squareIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_squareIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_squareIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_oddqIxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN9vfun_oddqIxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN9vfun_oddqIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN9vfun_oddqIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN9vfun_oddqIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_oddqIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_oddqIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_oddqIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_evenqIxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_evenqIxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_evenqIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_evenqIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_evenqIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_evenqIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_evenqIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_evenqIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_intpartIxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN12vfun_intpartIxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN12vfun_intpartIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN12vfun_intpartIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.trunc.f64(double) #0

declare i64 @_ZN12vfun_intpartIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_intpartIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_intpartIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_intpartIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_fracpartINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_fracpartINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_fracpartINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_fracpartINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_fracpartINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_fracpartINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_fracpartIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_fracpartIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_fracpartIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_fracpartIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_fracpartIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_fracpartIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare void @vdFrac(i64, double*, double*)

declare i64 @_ZN13vfun_fracpartIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_fracpartIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_fracpartIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_fracpartIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_ceilingIxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN12vfun_ceilingIxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN12vfun_ceilingIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN12vfun_ceilingIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.ceil.f64(double) #0

declare i64 @_ZN12vfun_ceilingIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_ceilingIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_ceilingIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_ceilingIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_floorIxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_floorIxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_floorIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_floorIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.floor.f64(double) #0

declare i64 @_ZN10vfun_floorIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_floorIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_floorIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_floorIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_roundIxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_roundIxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_roundIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_roundIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.nearbyint.f64(double) #0

declare i64 @_ZN10vfun_roundIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_roundIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_roundIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_roundIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_nnegIxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN9vfun_nnegIxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN9vfun_nnegIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN9vfun_nnegIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN9vfun_nnegIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_nnegIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_nnegIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_nnegIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_nposIxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN9vfun_nposIxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN9vfun_nposIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN9vfun_nposIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN9vfun_nposIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_nposIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_nposIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_nposIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_negIxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_negIxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_negIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_negIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_negIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_negIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_negIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_negIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_posIxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_posIxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_posIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_posIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_posIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_posIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_posIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_posIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_signINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_signINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_signINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_signINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_signINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_signINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_signIxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN9vfun_signIxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN9vfun_signIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN9vfun_signIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN9vfun_signIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_signIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_signIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_signIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_minusINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_minusINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_minusINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_minusINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_minusINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_minusINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_minusIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_minusIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_minusIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_minusIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_minusIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_minusIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_minusIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_minusIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_minusIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_minusIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN7vfun_reIdNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPdPKS2_RKxPSB_(double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN7vfun_reIdNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPdPKS2_RKxPSB_(double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN7vfun_reIdNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPdPKS2_RKxPSB_(double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN7vfun_reIdNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPdPKS2_RKxPSB_(double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN7vfun_reIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN7vfun_reIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN7vfun_reIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN7vfun_reIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN7vfun_reIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN7vfun_reIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN7vfun_reIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN7vfun_reIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN7vfun_reIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN7vfun_reIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN7vfun_imIdNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPdPKS2_RKxPSB_(double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN7vfun_imIdNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPdPKS2_RKxPSB_(double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN7vfun_imIdNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPdPKS2_RKxPSB_(double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN7vfun_imIdNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPdPKS2_RKxPSB_(double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN7vfun_imIxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN7vfun_imIxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN7vfun_imIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN7vfun_imIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN7vfun_imIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN7vfun_imIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN7vfun_imIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN7vfun_imIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_conjINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_conjINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_conjINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_conjINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_conjINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_conjINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare void @vzConj(i64, %"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare i64 @_ZN9vfun_conjIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_conjIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_conjIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_conjIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_conjIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_conjIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_conjIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_conjIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_conjIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_conjIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_argIdNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPdPKS2_RKxPSB_(double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_argIdNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPdPKS2_RKxPSB_(double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_argIdNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPdPKS2_RKxPSB_(double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_argIdNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPdPKS2_RKxPSB_(double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_argIxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_argIxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_argIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_argIxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKdRKxPS8_(i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_argIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_argIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_argIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_argIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_absIdNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPdPKS2_RKxPSB_(double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_absIdNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPdPKS2_RKxPSB_(double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_absIdNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPdPKS2_RKxPSB_(double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_absIdNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPdPKS2_RKxPSB_(double*, %"class.std::__1::complex"*, i64*, i64*)

declare void @vzAbs(i64, %"class.std::__1::complex"*, double*)

declare i64 @_ZN8vfun_absIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_absIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_absIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_absIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_absIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_absIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare void @vdAbs(i64, double*, double*)

declare i64 @_ZN8vfun_absIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_absIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_absIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_absIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_log10INSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_log10INSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_log10INSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_log10INSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_log10INSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_log10INSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.log10.f64(double) #0

declare void @vzLog10(i64, %"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare i64 @_ZN10vfun_log10IddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_log10IddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_log10IddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_log10IddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_log10IddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_log10IddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare void @vdLog10(i64, double*, double*)

declare i64 @_ZN10vfun_log10IxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_log10IxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_log10IxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_log10IxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare { i64, i64 } @lldiv(i64, i64)

declare i64 @_ZN9vfun_log2INSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_log2INSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_log2INSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_log2INSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_log2INSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_log2INSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.log2.f64(double) #0

declare i64 @_ZN9vfun_log2IddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_log2IddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_log2IddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_log2IddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_log2IddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_log2IddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_log2IxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_log2IxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_log2IxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_log2IxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_log1pINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_log1pINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_log1pINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_log1pINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_log1pINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_log1pINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op5unaryL8log1p_opINSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE47EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE(double, double)

declare i64 @_ZN10vfun_log1pIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_log1pIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_log1pIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_log1pIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_log1pIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_log1pIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare void @vdLog1p(i64, double*, double*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare void @vzLn(i64, %"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare i64 @_ZN8vfun_logIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare void @vdLn(i64, double*, double*)

declare i64 @_ZN10vfun_expm1INSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_expm1INSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_expm1INSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_expm1INSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_expm1INSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_expm1INSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare double @expm1(double)

declare { double, double } @_ZN3wrt9scalar_op5unaryL8expm1_opINSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE33EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE(double, double)

declare i64 @_ZN10vfun_expm1IddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_expm1IddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_expm1IddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_expm1IddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_expm1IddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_expm1IddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare void @vdExpm1(i64, double*, double*)

declare i64 @_ZN8vfun_expINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_expINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_expINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_expINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_expINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_expINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare void @vzExp(i64, %"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare i64 @_ZN8vfun_expIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_expIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_expIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_expIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_expIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_expIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare void @vdExp(i64, double*, double*)

declare i64 @_ZN10vfun_acothINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_acothINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_acothINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_acothINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_acothINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_acothINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op5unaryL8acoth_opINSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE6EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE(%"class.std::__1::complex"*)

declare i64 @_ZN10vfun_acothIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_acothIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_acothIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_acothIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_acothIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_acothIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_asechINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_asechINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_asechINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_asechINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_asechINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_asechINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op5unaryL8acosh_opINSt3__17complexIdEELNS_13runtime_flagsE0EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE4EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE(%"class.std::__1::complex"*)

declare { double, double } @_ZN3wrt9scalar_op5unaryL8asech_opINSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE11EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE(%"class.std::__1::complex"*)

declare i64 @_ZN10vfun_asechIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_asechIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_asechIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_asechIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_asechIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_asechIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare double @acosh(double)

declare i64 @_ZN10vfun_acschINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_acschINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_acschINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_acschINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_acschINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_acschINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op5unaryL8acsch_opINSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXsr28is_floating_point_or_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE8EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE(%"class.std::__1::complex"*)

declare i64 @_ZN10vfun_acschIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_acschIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_acschIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_acschIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_acschIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_acschIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare double @asinh(double)

declare i64 @_ZN10vfun_atanhINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atanhINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atanhINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atanhINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atanhINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atanhINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op5unaryL8atanh_opINSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE15EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE(%"class.std::__1::complex"*)

declare i64 @_ZN10vfun_atanhIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atanhIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atanhIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atanhIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atanhIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atanhIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare void @vdAtanh(i64, double*, double*)

declare i64 @_ZN10vfun_acoshINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_acoshINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_acoshINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_acoshINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_acoshINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_acoshINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare void @vzAcosh(i64, %"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare { double, double } @_ZN3wrt9scalar_op5unaryL8acosh_opINSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE4EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE(%"class.std::__1::complex"*)

declare i64 @_ZN10vfun_acoshIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_acoshIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_acoshIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_acoshIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_acoshIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_acoshIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare void @vdAcosh(i64, double*, double*)

declare i64 @_ZN10vfun_asinhINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_asinhINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_asinhINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_asinhINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_asinhINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_asinhINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op5unaryL8asinh_opINSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE13EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE(%"class.std::__1::complex"*)

declare i64 @_ZN10vfun_asinhIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_asinhIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_asinhIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_asinhIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_asinhIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_asinhIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare void @vdAsinh(i64, double*, double*)

declare i64 @_ZN9vfun_acotINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_acotINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_acotINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_acotINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_acotINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_acotINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_acotIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_acotIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_acotIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_acotIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_acotIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_acotIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_asecINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_asecINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_asecINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_asecINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_asecINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_asecINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_asecIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_asecIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_asecIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_asecIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_asecIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_asecIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare double @acos(double)

declare i64 @_ZN9vfun_acscINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_acscINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_acscINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_acscINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_acscINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_acscINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_acscIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_acscIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_acscIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_acscIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_acscIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_acscIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_atanINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_atanINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_atanINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_atanINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_atanINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_atanINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op5unaryL7atan_opINSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE14EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE(%"class.std::__1::complex"*)

declare i64 @_ZN9vfun_atanIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_atanIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_atanIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_atanIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_atanIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_atanIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare void @vdAtan(i64, double*, double*)

declare i64 @_ZN9vfun_acosINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_acosINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_acosINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_acosINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_acosINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_acosINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_acosIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_acosIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_acosIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_acosIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_acosIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_acosIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare void @vdAcos(i64, double*, double*)

declare i64 @_ZN9vfun_asinINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_asinINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_asinINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_asinINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_asinINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_asinINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_asinIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_asinIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_asinIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_asinIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_asinIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_asinIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare void @vdAsin(i64, double*, double*)

declare i64 @_ZN9vfun_cothINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_cothINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_cothINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_cothINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_cothINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_cothINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op5unaryL7coth_opINSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXsr28is_floating_point_or_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE26EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE(%"class.std::__1::complex"*)

declare i64 @_ZN9vfun_cothIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_cothIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_cothIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_cothIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_cothIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_cothIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_sechINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_sechINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_sechINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_sechINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_sechINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_sechINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_sechIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_sechIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_sechIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_sechIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_sechIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_sechIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_cschINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_cschINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_cschINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_cschINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_cschINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_cschINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_cschIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_cschIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_cschIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_cschIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_cschIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_cschIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_tanhINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_tanhINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_tanhINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_tanhINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_tanhINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_tanhINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare void @vzTanh(i64, %"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare i64 @_ZN9vfun_tanhIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_tanhIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_tanhIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_tanhIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_tanhIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_tanhIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare void @vdTanh(i64, double*, double*)

declare i64 @_ZN9vfun_coshINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_coshINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_coshINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_coshINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_coshINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_coshINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare void @vzCosh(i64, %"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare i64 @_ZN9vfun_coshIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_coshIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_coshIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_coshIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_coshIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_coshIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare void @vdCosh(i64, double*, double*)

declare i64 @_ZN9vfun_sinhINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_sinhINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_sinhINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_sinhINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_sinhINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_sinhINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare void @vzSinh(i64, %"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare i64 @_ZN9vfun_sinhIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_sinhIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_sinhIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_sinhIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_sinhIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_sinhIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare void @vdSinh(i64, double*, double*)

declare i64 @_ZN8vfun_cotINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_cotINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_cotINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_cotINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_cotINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_cotINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op5unaryL6cot_opINSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXsr28is_floating_point_or_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE25EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE(double, double)

declare i64 @_ZN8vfun_cotIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_cotIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_cotIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_cotIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_cotIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_cotIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_secINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_secINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_secINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_secINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_secINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_secINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op5unaryL6sec_opINSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXsr28is_floating_point_or_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE66EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE(double, double)

declare i64 @_ZN8vfun_secIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_secIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_secIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_secIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_secIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_secIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_cscINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_cscINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_cscINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_cscINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_cscINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_cscINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_cscIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_cscIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_cscIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_cscIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_cscIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_cscIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_tanINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_tanINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_tanINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_tanINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_tanINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_tanINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare void @vzTan(i64, %"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare i64 @_ZN8vfun_tanIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_tanIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_tanIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_tanIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_tanIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_tanIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare void @vdTan(i64, double*, double*)

declare i64 @_ZN8vfun_cosINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_cosINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_cosINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_cosINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_cosINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_cosINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare void @vzCos(i64, %"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare i64 @_ZN8vfun_cosIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_cosIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_cosIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_cosIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_cosIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_cosIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare void @vdCos(i64, double*, double*)

declare i64 @_ZN8vfun_sinINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_sinINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_sinINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_sinINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_sinINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_sinINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare void @vzSin(i64, %"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare i64 @_ZN8vfun_sinIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_sinIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_sinIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_sinIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_sinIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_sinIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare void @vdSin(i64, double*, double*)

declare i64 @machfunc_i_copy(i8*, i8*, i64, i64*, i32)

declare i64 @_ZN9vfun_copyIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_copyIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_copyIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_copyIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @machfunc_d_copy(i8*, i8*, i64, i64*, i32)

declare i64 @_ZN9vfun_copyIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_copyIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_copyIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_copyIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_copyIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_copyIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @machfunc_c_copy(i8*, i8*, i64, i64*, i32)

declare i64 @_ZN9vfun_copyINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_copyINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_copyINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_copyINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_copyINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_copyINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @machfunc_i_zero(i8*, i8*, i64, i64*, i32)

declare i64 @_ZN9vfun_zeroIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_zeroIxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_zeroIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_zeroIxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxRS6_S7_(i64*, i64*, i64*, i64*)

declare i64 @machfunc_d_zero(i8*, i8*, i64, i64*, i32)

declare i64 @_ZN9vfun_zeroIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_zeroIddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_zeroIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_zeroIddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_zeroIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_zeroIddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdRKxPS8_(double*, double*, i64*, i64*)

declare i64 @machfunc_c_zero(i8*, i8*, i64, i64*, i32)

declare i64 @_ZN9vfun_zeroINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_zeroINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_zeroINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_zeroINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_zeroINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_zeroINSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i32 @ReturnRangeType1(i32, i32, i32*)

declare i32 @ReturnType1(i32, i32)

declare i64 @machfunc_ii_dot(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZN8vfun_dotIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_dotIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_dotIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_dotIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN3wrt9scalar_op6binaryL7plus_opIxxLNS_13runtime_flagsE0EEENSt3__19enable_ifIXaasr11is_integralIT_EE5valuesr11is_integralIT0_EE5valueENS1_6detail7op_infoILNS8_2opE19EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE.30(i64*, i64*)

declare i64 @_ZN3wrt9scalar_op6binaryL7plus_opIxxLNS_13runtime_flagsE1EEENSt3__19enable_ifIXaasr11is_integralIT_EE5valuesr11is_integralIT0_EE5valueENS1_6detail7op_infoILNS8_2opE19EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE.31(i64*, i64*)

declare void @.omp_outlined..32(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (i64*, i64*, i64*, i64*, i64*)**, i64**, i64**, i64**, i64**, i64 (i64*, i64*)**)

declare i64 @machfunc_ii_plus(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_id_plus(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_ic_plus(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL16machfunc_di_plusPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_dd_plus(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_dc_plus(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL16machfunc_ci_plusPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL16machfunc_cd_plusPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_cc_plus(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_ii_subtract(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_id_subtract(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_ic_subtract(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_di_subtract(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_dd_subtract(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_dc_subtract(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_ci_subtract(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_cd_subtract(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_cc_subtract(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_ii_times(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_id_times(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_ic_times(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_di_times(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_dd_times(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_dc_times(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_ci_times(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_cd_times(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_cc_times(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_id_div(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_ic_div(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_di_div(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_dd_div(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_dc_div(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_ci_div(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_cd_div(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_cc_div(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_ii_mod(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_id_modPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_ic_modPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_di_modPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_dd_mod(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_dc_modPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_ci_modPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_cd_modPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_cc_mod(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_ii_quotient(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL20machfunc_id_quotientPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL20machfunc_di_quotientPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_dd_quotient(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_ii_pow(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_id_pow(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_ic_pow(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_di_pow(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_dd_pow(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_dc_pow(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_ci_pow(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_cd_pow(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_cc_pow(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_id_logPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_ic_logPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_di_logPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_dd_logPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_dc_logPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_ci_logPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_cd_logPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL15machfunc_cc_logPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL17machfunc_id_atan2PvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL17machfunc_ic_atan2PvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL17machfunc_di_atan2PvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL17machfunc_dd_atan2PvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL17machfunc_dc_atan2PvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL17machfunc_ci_atan2PvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL17machfunc_cd_atan2PvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL17machfunc_cc_atan2PvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_ii_bitand(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_ii_bitor(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_ii_bitxor(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL16machfunc_dd_chopPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL16machfunc_cd_chopPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_ii_abserr(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL18machfunc_id_abserrPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL18machfunc_ic_abserrPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL18machfunc_di_abserrPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_dd_abserr(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL18machfunc_dc_abserrPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL18machfunc_ci_abserrPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL18machfunc_cd_abserrPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_cc_abserr(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL18machfunc_id_relerrPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL18machfunc_ic_relerrPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL18machfunc_di_relerrPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_dd_relerr(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL18machfunc_dc_relerrPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL18machfunc_ci_relerrPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL18machfunc_cd_relerrPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_cc_relerr(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_ii_maxabs(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_dd_maxabs(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_cc_maxabs(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_ii_intexp(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_ii_intlen(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_ii_bitshiftleft(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_di_bitshiftleft(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_ci_bitshiftleft(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_ii_bitshiftright(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_di_bitshiftright(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @machfunc_ci_bitshiftright(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL19machfunc_ii_unitizePvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL19machfunc_dd_unitizePvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZL19machfunc_cd_unitizePvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZN12vfun_unitizeIxNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPxPKS2_PKdRKxPSD_(i64*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPxPKS2_PKdRKxPSD_(i64*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare void @.omp_outlined..12.38(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (i64*, %"class.std::__1::complex"*, double*, i64*, i64*)**, i64**, %"class.std::__1::complex"**, double**, i64**)

declare i64 @_ZN3wrt9scalar_op6binaryL10unitize_opINSt3__17complexIdEEdLNS_13runtime_flagsE1EEENS3_9enable_ifIXntsr10is_complexIT0_EE5valueENS1_6detail7op_infoILNS9_2opE27EN7rt_typeIT_E4typeENSC_IS8_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(%"class.std::__1::complex"*, double*)

declare i64 @_ZN12vfun_unitizeIxNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPxPKS2_PKdRKxPSD_(i64*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPxPKS2_PKdRKxPSD_(i64*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPxPKS2_PKdRKxPSD_(i64*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPxPKS2_PKdRKxPSD_(i64*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPxPKS2_PKdRKxPSD_(i64*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPxPKS2_PKdRKxPSD_(i64*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKdS7_RKxPS8_(i64*, double*, double*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKdS7_RKxPS8_(i64*, double*, double*, i64*, i64*)

declare void @.omp_outlined..36(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (i64*, double*, double*, i64*, i64*)**, i64**, double**, double**, i64**)

declare i64 @_ZN3wrt9scalar_op6binaryL10unitize_opIddLNS_13runtime_flagsE1EEENSt3__19enable_ifIXntsr10is_complexIT0_EE5valueENS1_6detail7op_infoILNS7_2opE27EN7rt_typeIT_E4typeENSA_IS6_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE(double*, double*)

declare i64 @_ZN12vfun_unitizeIxddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKdS7_RKxPS8_(i64*, double*, double*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPxPKdS7_RKxPS8_(i64*, double*, double*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPxPKdS7_RKxPS8_(i64*, double*, double*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKdS7_RKxPS8_(i64*, double*, double*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPxPKdS7_RKxPS8_(i64*, double*, double*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPxPKdS7_RKxPS8_(i64*, double*, double*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare void @.omp_outlined..14.42(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (i64*, i64*, i64*, i64*, i64*)**, i64**, i64**, i64**, i64**)

declare i64 @_ZN12vfun_unitizeIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_unitizeIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN19vfun_bit_shiftrightINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN19vfun_bit_shiftrightINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN19vfun_bit_shiftrightINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare void @.omp_outlined..24(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)**, %"class.std::__1::complex"**, %"class.std::__1::complex"**, i64**, i64**)

declare { double, double } @_ZN3wrt9scalar_op6binaryL17bit_shiftright_opINSt3__17complexIdEExLNS_13runtime_flagsE0EEENS3_9enable_ifIXaaoosr11is_integralIT_EE5valuesr28is_floating_point_or_complexIS8_EE5valuesr18is_signed_integralIT0_EE5valueENS1_6detail7op_infoILNSA_2opE6EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(%"class.std::__1::complex"*, i64)

declare { double, double } @_ZN3wrt9scalar_op6binaryL17bit_shiftright_opINSt3__17complexIdEExLNS_13runtime_flagsE1EEENS3_9enable_ifIXaaoosr11is_integralIT_EE5valuesr28is_floating_point_or_complexIS8_EE5valuesr18is_signed_integralIT0_EE5valueENS1_6detail7op_infoILNSA_2opE6EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(%"class.std::__1::complex"*, i64)

declare i64 @_ZN19vfun_bit_shiftrightINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN19vfun_bit_shiftrightINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN19vfun_bit_shiftrightINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN19vfun_bit_shiftrightINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN19vfun_bit_shiftrightINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN19vfun_bit_shiftrightINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN19vfun_bit_shiftrightINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN19vfun_bit_shiftrightINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN19vfun_bit_shiftrightINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN19vfun_bit_shiftrightIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN19vfun_bit_shiftrightIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN19vfun_bit_shiftrightIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare void @.omp_outlined..18(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (double*, double*, i64*, i64*, i64*)**, double**, double**, i64**, i64**)

declare i64 @_ZN19vfun_bit_shiftrightIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN19vfun_bit_shiftrightIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN19vfun_bit_shiftrightIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN19vfun_bit_shiftrightIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN19vfun_bit_shiftrightIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN19vfun_bit_shiftrightIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN19vfun_bit_shiftrightIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN19vfun_bit_shiftrightIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN19vfun_bit_shiftrightIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN19vfun_bit_shiftrightIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN19vfun_bit_shiftrightIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN3wrt9scalar_op6binaryL17bit_shiftright_opIxxLNS_13runtime_flagsE1EEENSt3__19enable_ifIXaaoosr11is_integralIT_EE5valuesr28is_floating_point_or_complexIS6_EE5valuesr18is_signed_integralIT0_EE5valueENS1_6detail7op_infoILNS8_2opE6EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE(i64*, i64)

declare i64 @_ZN19vfun_bit_shiftrightIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN19vfun_bit_shiftrightIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN19vfun_bit_shiftrightIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN19vfun_bit_shiftrightIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN19vfun_bit_shiftrightIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN19vfun_bit_shiftrightIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN18vfun_bit_shiftleftIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_intlenIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_intlenIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_intlenIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_intlenIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_intlenIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_intlenIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_intlenIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_intlenIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_intexpIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_intexpIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN3wrt9scalar_op6binaryL9intexp_opIxxLNS_13runtime_flagsE1EEENSt3__19enable_ifIXaasr11is_integralIT_EE5valuesr11is_integralIT0_EE5valueENS1_6detail7op_infoILNS8_2opE11EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE(i64*, i64*)

declare i64 @_ZN11vfun_intexpIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_intexpIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_intexpIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_intexpIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_intexpIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_intexpIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIdNSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPdPKS2_SA_RKxPSB_(double*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIdNSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPdPKS2_SA_RKxPSB_(double*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare void @.omp_outlined..32.43(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (double*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)**, double**, %"class.std::__1::complex"**, %"class.std::__1::complex"**, i64**)

declare i64 @_ZN11vfun_maxabsIdNSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPdPKS2_SA_RKxPSB_(double*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIdNSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPdPKS2_SA_RKxPSB_(double*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIdNSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPdPKS2_SA_RKxPSB_(double*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIdNSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPdPKS2_SA_RKxPSB_(double*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIdNSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPdPKS2_SA_RKxPSB_(double*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIdNSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPdPKS2_SA_RKxPSB_(double*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare void @.omp_outlined..20(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (double*, double*, double*, i64*, i64*)**, double**, double**, double**, i64**)

declare i64 @_ZN11vfun_maxabsIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_maxabsIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdNSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPdPKS2_SA_RKxPSB_(double*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdNSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPdPKS2_SA_RKxPSB_(double*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdNSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPdPKS2_SA_RKxPSB_(double*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdNSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPdPKS2_SA_RKxPSB_(double*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdNSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPdPKS2_SA_RKxPSB_(double*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdNSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPdPKS2_SA_RKxPSB_(double*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdNSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPdPKS2_SA_RKxPSB_(double*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdNSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPdPKS2_SA_RKxPSB_(double*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPdPKS2_PKdRKxPSD_(double*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPdPKS2_PKdRKxPSD_(double*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPdPKS2_PKdRKxPSD_(double*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare void @.omp_outlined..10.44(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (double*, %"class.std::__1::complex"*, double*, i64*, i64*)**, double**, %"class.std::__1::complex"**, double**, i64**)

declare i64 @_ZN11vfun_relerrIdNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPdPKS2_PKdRKxPSD_(double*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPdPKS2_PKdRKxPSD_(double*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPdPKS2_PKdRKxPSD_(double*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPdPKS2_PKdRKxPSD_(double*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPdPKS2_PKdRKxPSD_(double*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPdPKS2_PKdRKxPSD_(double*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPdPKS2_PKdRKxPSD_(double*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPdPKS2_PKdRKxPSD_(double*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPdPKS2_PKdRKxPSD_(double*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdNSt3__17complexIdEExE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPdPKS2_PKxRSB_SC_(double*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdNSt3__17complexIdEExE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPdPKS2_PKxRSB_SC_(double*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare void @.omp_outlined..8.45(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (double*, %"class.std::__1::complex"*, i64*, i64*, i64*)**, double**, %"class.std::__1::complex"**, i64**, i64**)

declare i64 @_ZN11vfun_relerrIdNSt3__17complexIdEExE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPdPKS2_PKxRSB_SC_(double*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdNSt3__17complexIdEExE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPdPKS2_PKxRSB_SC_(double*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdNSt3__17complexIdEExE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPdPKS2_PKxRSB_SC_(double*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdNSt3__17complexIdEExE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPdPKS2_PKxRSB_SC_(double*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdNSt3__17complexIdEExE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPdPKS2_PKxRSB_SC_(double*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdNSt3__17complexIdEExE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPdPKS2_PKxRSB_SC_(double*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIddNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPdPKdPKS2_RKxPSD_(double*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIddNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPdPKdPKS2_RKxPSD_(double*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIddNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPdPKdPKS2_RKxPSD_(double*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare void @.omp_outlined..6.46(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (double*, double*, %"class.std::__1::complex"*, i64*, i64*)**, double**, double**, %"class.std::__1::complex"**, i64**)

declare i64 @_ZN11vfun_relerrIddNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPdPKdPKS2_RKxPSD_(double*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIddNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPdPKdPKS2_RKxPSD_(double*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIddNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPdPKdPKS2_RKxPSD_(double*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIddNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPdPKdPKS2_RKxPSD_(double*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIddNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPdPKdPKS2_RKxPSD_(double*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIddNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPdPKdPKS2_RKxPSD_(double*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIddNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPdPKdPKS2_RKxPSD_(double*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIddNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPdPKdPKS2_RKxPSD_(double*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIddNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPdPKdPKS2_RKxPSD_(double*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdxNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPdPKxPKS2_RS9_SA_(double*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdxNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPdPKxPKS2_RS9_SA_(double*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare void @.omp_outlined..4.47(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (double*, i64*, %"class.std::__1::complex"*, i64*, i64*)**, double**, i64**, %"class.std::__1::complex"**, i64**)

declare i64 @_ZN11vfun_relerrIdxNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPdPKxPKS2_RS9_SA_(double*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdxNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPdPKxPKS2_RS9_SA_(double*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdxNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPdPKxPKS2_RS9_SA_(double*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdxNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPdPKxPKS2_RS9_SA_(double*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdxNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPdPKxPKS2_RS9_SA_(double*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdxNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPdPKxPKS2_RS9_SA_(double*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare void @.omp_outlined..16(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (double*, i64*, double*, i64*, i64*)**, double**, i64**, double**, i64**)

declare i64 @_ZN11vfun_relerrIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_relerrIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdNSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPdPKS2_SA_RKxPSB_(double*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdNSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPdPKS2_SA_RKxPSB_(double*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdNSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPdPKS2_SA_RKxPSB_(double*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdNSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPdPKS2_SA_RKxPSB_(double*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdNSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPdPKS2_SA_RKxPSB_(double*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdNSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPdPKS2_SA_RKxPSB_(double*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdNSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPdPKS2_SA_RKxPSB_(double*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdNSt3__17complexIdEES2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPdPKS2_SA_RKxPSB_(double*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPdPKS2_PKdRKxPSD_(double*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPdPKS2_PKdRKxPSD_(double*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPdPKS2_PKdRKxPSD_(double*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare double @_ZN3wrt9scalar_op6binaryL9abserr_opINSt3__17complexIdEEdLNS_13runtime_flagsE1EEENS1_6detail7op_infoILNS7_2opE1EN7rt_typeIT_E4typeENSA_IT0_E4typeEE11result_typeERKNSH_19first_argument_typeERKNSH_20second_argument_typeE(double, double, double)

declare i64 @_ZN11vfun_abserrIdNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPdPKS2_PKdRKxPSD_(double*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPdPKS2_PKdRKxPSD_(double*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPdPKS2_PKdRKxPSD_(double*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPdPKS2_PKdRKxPSD_(double*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPdPKS2_PKdRKxPSD_(double*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPdPKS2_PKdRKxPSD_(double*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPdPKS2_PKdRKxPSD_(double*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPdPKS2_PKdRKxPSD_(double*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdNSt3__17complexIdEEdE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPdPKS2_PKdRKxPSD_(double*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdNSt3__17complexIdEExE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPdPKS2_PKxRSB_SC_(double*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdNSt3__17complexIdEExE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPdPKS2_PKxRSB_SC_(double*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdNSt3__17complexIdEExE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPdPKS2_PKxRSB_SC_(double*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdNSt3__17complexIdEExE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPdPKS2_PKxRSB_SC_(double*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdNSt3__17complexIdEExE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPdPKS2_PKxRSB_SC_(double*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdNSt3__17complexIdEExE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPdPKS2_PKxRSB_SC_(double*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdNSt3__17complexIdEExE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPdPKS2_PKxRSB_SC_(double*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdNSt3__17complexIdEExE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPdPKS2_PKxRSB_SC_(double*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIddNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPdPKdPKS2_RKxPSD_(double*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIddNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPdPKdPKS2_RKxPSD_(double*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIddNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPdPKdPKS2_RKxPSD_(double*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare double @_ZN3wrt9scalar_op6binaryL9abserr_opIdNSt3__17complexIdEELNS_13runtime_flagsE1EEENS1_6detail7op_infoILNS7_2opE1EN7rt_typeIT_E4typeENSA_IT0_E4typeEE11result_typeERKNSH_19first_argument_typeERKNSH_20second_argument_typeE(double, double, double)

declare i64 @_ZN11vfun_abserrIddNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPdPKdPKS2_RKxPSD_(double*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIddNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPdPKdPKS2_RKxPSD_(double*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIddNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPdPKdPKS2_RKxPSD_(double*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIddNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPdPKdPKS2_RKxPSD_(double*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIddNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPdPKdPKS2_RKxPSD_(double*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIddNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPdPKdPKS2_RKxPSD_(double*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIddNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPdPKdPKS2_RKxPSD_(double*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIddNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPdPKdPKS2_RKxPSD_(double*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIddNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPdPKdPKS2_RKxPSD_(double*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdxNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPdPKxPKS2_RS9_SA_(double*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdxNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPdPKxPKS2_RS9_SA_(double*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdxNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPdPKxPKS2_RS9_SA_(double*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdxNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPdPKxPKS2_RS9_SA_(double*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdxNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPdPKxPKS2_RS9_SA_(double*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdxNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPdPKxPKS2_RS9_SA_(double*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdxNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPdPKxPKS2_RS9_SA_(double*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdxNSt3__17complexIdEEE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPdPKxPKS2_RS9_SA_(double*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

; Function Attrs: nounwind readnone speculatable
declare { i64, i1 } @llvm.ssub.with.overflow.i64(i64, i64) #0

declare i64 @_ZN11vfun_abserrIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_abserrIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_chopINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN9vfun_chopINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN9vfun_chopINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare void @.omp_outlined..28.48(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)**, %"class.std::__1::complex"**, %"class.std::__1::complex"**, double**, i64**)

declare { double, double } @_ZN3wrt9scalar_op6binaryL7chop_opINSt3__17complexIdEEdLNS_13runtime_flagsE1EEENS3_9enable_ifIXaasr10is_complexIT_EE5valuesr17is_floating_pointIT0_EE5valueENS1_6detail7op_infoILNSA_2opE8EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(%"class.std::__1::complex"*, double*)

declare i64 @_ZN9vfun_chopINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN9vfun_chopINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN9vfun_chopINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN9vfun_chopINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN9vfun_chopINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN9vfun_chopINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN9vfun_chopINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN9vfun_chopINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN9vfun_chopINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN9vfun_chopIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_chopIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_chopIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_chopIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_chopIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_chopIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_chopIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_chopIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_chopIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_chopIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_chopIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_chopIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN12vfun_bit_xorIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_bit_xorIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_bit_xorIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_bit_xorIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_bit_xorIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_bit_xorIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_bit_xorIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_bit_xorIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_bit_orIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_bit_orIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_bit_orIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_bit_orIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_bit_orIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_bit_orIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_bit_orIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_bit_orIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_bit_andIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_bit_andIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_bit_andIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_bit_andIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_bit_andIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_bit_andIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_bit_andIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN12vfun_bit_andIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare void @.omp_outlined..30(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)**, %"class.std::__1::complex"**, %"class.std::__1::complex"**, %"class.std::__1::complex"**, i64**)

declare { double, double } @_ZN3wrt9scalar_op6binaryL8atan2_opINSt3__17complexIdEES5_LNS_13runtime_flagsE0EEENS3_9enable_ifIXoosr10is_complexIT_EE5valuesr10is_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE2EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(double, double, double, double)

declare { double, double } @_ZN3wrt9scalar_op6binaryL8atan2_opINSt3__17complexIdEES5_LNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr10is_complexIT_EE5valuesr10is_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE2EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(double, double, double, double)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL8atan2_opINSt3__17complexIdEEdLNS_13runtime_flagsE0EEENS3_9enable_ifIXoosr10is_complexIT_EE5valuesr10is_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE2EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(double, double, double)

declare { double, double } @_ZN3wrt9scalar_op6binaryL8atan2_opINSt3__17complexIdEEdLNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr10is_complexIT_EE5valuesr10is_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE2EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(double, double, double)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL8atan2_opINSt3__17complexIdEExLNS_13runtime_flagsE0EEENS3_9enable_ifIXoosr10is_complexIT_EE5valuesr10is_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE2EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(double, double, i64)

declare { double, double } @_ZN3wrt9scalar_op6binaryL8atan2_opINSt3__17complexIdEExLNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr10is_complexIT_EE5valuesr10is_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE2EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(double, double, i64)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare void @.omp_outlined..26(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)**, %"class.std::__1::complex"**, double**, %"class.std::__1::complex"**, i64**)

declare { double, double } @_ZN3wrt9scalar_op6binaryL8atan2_opIdNSt3__17complexIdEELNS_13runtime_flagsE0EEENS3_9enable_ifIXoosr10is_complexIT_EE5valuesr10is_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE2EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(double, double, double)

declare { double, double } @_ZN3wrt9scalar_op6binaryL8atan2_opIdNSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr10is_complexIT_EE5valuesr10is_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE2EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(double, double, double)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare void @vdAtan2(i64, double*, double*, double*)

declare i64 @_ZN10vfun_atan2IdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare void @.omp_outlined..22(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)**, %"class.std::__1::complex"**, i64**, %"class.std::__1::complex"**, i64**)

declare { double, double } @_ZN3wrt9scalar_op6binaryL8atan2_opIxNSt3__17complexIdEELNS_13runtime_flagsE0EEENS3_9enable_ifIXoosr10is_complexIT_EE5valuesr10is_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE2EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(i64, double, double)

declare { double, double } @_ZN3wrt9scalar_op6binaryL8atan2_opIxNSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr10is_complexIT_EE5valuesr10is_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE2EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(i64, double, double)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2INSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_atan2IdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6log_opINSt3__17complexIdEES5_LNS_13runtime_flagsE0EEENS3_9enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE13EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(double, double, double, double)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6log_opINSt3__17complexIdEES5_LNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE13EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(double, double, double, double)

declare { double, double } @_ZN3wrt9scalar_op6binaryL9divide_opINSt3__17complexIdEES5_LNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE9EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE.50(%"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6log_opINSt3__17complexIdEEdLNS_13runtime_flagsE0EEENS3_9enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE13EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(double, double, double)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6log_opINSt3__17complexIdEEdLNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE13EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(double, double, double)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6log_opINSt3__17complexIdEExLNS_13runtime_flagsE0EEENS3_9enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE13EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(double, double, i64)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6log_opINSt3__17complexIdEExLNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE13EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(double, double, i64)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6log_opIdNSt3__17complexIdEELNS_13runtime_flagsE0EEENS3_9enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE13EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(double, double, double)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6log_opIdNSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE13EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(double, double, double)

declare i64 @_ZN8vfun_logINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare double @_ZN3wrt9scalar_op6binaryL6log_opIddLNS_13runtime_flagsE1EEENSt3__19enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNS8_2opE13EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE(double, double*)

declare i64 @_ZN8vfun_logIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_logIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_logIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_logIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_logIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_logIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_logIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_logIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_logIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_logIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_logIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_logIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6log_opIxNSt3__17complexIdEELNS_13runtime_flagsE0EEENS3_9enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE13EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(i64, double, double)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6log_opIxNSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE13EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(i64, double, double)

declare i64 @_ZN8vfun_logINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_logIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_logIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6pow_opINSt3__17complexIdEES5_LNS_13runtime_flagsE1EEENS3_9enable_ifIXaasr10is_complexIT_EE5valuesr10is_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE20EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(double, double, %"class.std::__1::complex"*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6pow_opINSt3__17complexIdEEdLNS_13runtime_flagsE0EEENS3_9enable_ifIXaasr10is_complexIT_EE5valuesr17is_floating_pointIT0_EE5valueENS1_6detail7op_infoILNSA_2opE20EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(double, double, double*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6pow_opINSt3__17complexIdEEdLNS_13runtime_flagsE1EEENS3_9enable_ifIXaasr10is_complexIT_EE5valuesr17is_floating_pointIT0_EE5valueENS1_6detail7op_infoILNSA_2opE20EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(double, double, double*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6pow_opINSt3__17complexIdEExLNS_13runtime_flagsE0EEENS3_9enable_ifIXaasr10is_complexIT_EE5valuesr11is_integralIT0_EE5valueENS1_6detail7op_infoILNSA_2opE20EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(%"class.std::__1::complex"*, i64)

declare double @_ZN3wrt9scalar_op6binaryL9pow_xi_opIdxLNS_13runtime_flagsE0EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(double, i64)

declare { double, double } @_ZN3wrt9scalar_op6binaryL9pow_xi_opINSt3__17complexIdEExLNS_13runtime_flagsE0EEENS3_9enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS9_RKS8_(%"class.std::__1::complex"*, i64)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6pow_opINSt3__17complexIdEExLNS_13runtime_flagsE1EEENS3_9enable_ifIXaasr10is_complexIT_EE5valuesr11is_integralIT0_EE5valueENS1_6detail7op_infoILNSA_2opE20EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(%"class.std::__1::complex"*, i64*)

declare double @_ZN3wrt9scalar_op6binaryL6pow_opIdxLNS_13runtime_flagsE1EEENSt3__19enable_ifIXaasr17is_floating_pointIT_EE5valuesr11is_integralIT0_EE5valueENS1_6detail7op_infoILNS8_2opE20EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE(double*, i64)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6pow_opIdNSt3__17complexIdEELNS_13runtime_flagsE0EEENS3_9enable_ifIXaaoosr11is_integralIT_EE5valuesr17is_floating_pointIS8_EE5valuesr10is_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE20EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(double, %"class.std::__1::complex"*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6pow_opIdNSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXaaoosr11is_integralIT_EE5valuesr17is_floating_pointIS8_EE5valuesr10is_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE20EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(double, %"class.std::__1::complex"*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_powIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_powIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_powIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_powIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_powIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_powIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_powIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_powIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_powIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_powIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_powIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6pow_opIxNSt3__17complexIdEELNS_13runtime_flagsE0EEENS3_9enable_ifIXaaoosr11is_integralIT_EE5valuesr17is_floating_pointIS8_EE5valuesr10is_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE20EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(i64, %"class.std::__1::complex"*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6pow_opIxNSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXaaoosr11is_integralIT_EE5valuesr17is_floating_pointIS8_EE5valuesr10is_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE20EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(i64, %"class.std::__1::complex"*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_powIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_powIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_powIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN3wrt9scalar_op6binaryL6pow_opIxxLNS_13runtime_flagsE0EEENSt3__19enable_ifIXaasr18is_signed_integralIT_EE5valuesr11is_integralIT0_EE5valueENS1_6detail7op_infoILNS8_2opE20EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE(i64, i64*)

declare i64 @_ZN3wrt9scalar_op6binaryL6pow_opIxxLNS_13runtime_flagsE1EEENSt3__19enable_ifIXaasr18is_signed_integralIT_EE5valuesr11is_integralIT0_EE5valueENS1_6detail7op_infoILNS8_2opE20EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE(i64, i64*)

declare i64 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIxxLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i64, i64)

declare i64 @_ZN8vfun_powIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_powIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_powIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_powIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_powIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_powIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKdS7_RKxPS8_(i64*, double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKdS7_RKxPS8_(i64*, double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKdS7_RKxPS8_(i64*, double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPxPKdS7_RKxPS8_(i64*, double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPxPKdS7_RKxPS8_(i64*, double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKdS7_RKxPS8_(i64*, double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPxPKdS7_RKxPS8_(i64*, double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPxPKdS7_RKxPS8_(i64*, double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxdxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKdPKxRS8_S9_(i64*, double*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxdxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKdPKxRS8_S9_(i64*, double*, i64*, i64*, i64*)

declare void @.omp_outlined..2.51(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (i64*, double*, i64*, i64*, i64*)**, i64**, double**, i64**, i64**)

declare i64 @_ZN13vfun_quotientIxdxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKdPKxRS8_S9_(i64*, double*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxdxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPxPKdPKxRS8_S9_(i64*, double*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxdxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPxPKdPKxRS8_S9_(i64*, double*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxdxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKdPKxRS8_S9_(i64*, double*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxdxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPxPKdPKxRS8_S9_(i64*, double*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxdxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPxPKdPKxRS8_S9_(i64*, double*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxPKdRS6_S7_(i64*, i64*, double*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxPKdRS6_S7_(i64*, i64*, double*, i64*, i64*)

declare void @.omp_outlined..52(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (i64*, i64*, double*, i64*, i64*)**, i64**, i64**, double**, i64**)

declare i64 @_ZN13vfun_quotientIxxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxPKdRS6_S7_(i64*, i64*, double*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPxPKxPKdRS6_S7_(i64*, i64*, double*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPxPKxPKdRS6_S7_(i64*, i64*, double*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxPKdRS6_S7_(i64*, i64*, double*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPxPKxPKdRS6_S7_(i64*, i64*, double*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPxPKxPKdRS6_S7_(i64*, i64*, double*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_quotientIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6mod_opINSt3__17complexIdEES5_LNS_13runtime_flagsE0EEENS3_9enable_ifIXoosr10is_complexIT_EE5valuesr10is_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE18EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(%"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6mod_opINSt3__17complexIdEES5_LNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr10is_complexIT_EE5valuesr10is_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE18EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(%"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6mod_opINSt3__17complexIdEEdLNS_13runtime_flagsE0EEENS3_9enable_ifIXoosr10is_complexIT_EE5valuesr10is_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE18EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(%"class.std::__1::complex"*, double)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6mod_opINSt3__17complexIdEEdLNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr10is_complexIT_EE5valuesr10is_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE18EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(%"class.std::__1::complex"*, double)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6mod_opINSt3__17complexIdEExLNS_13runtime_flagsE0EEENS3_9enable_ifIXoosr10is_complexIT_EE5valuesr10is_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE18EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(%"class.std::__1::complex"*, i64)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6mod_opINSt3__17complexIdEExLNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr10is_complexIT_EE5valuesr10is_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE18EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(%"class.std::__1::complex"*, i64)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6mod_opIdNSt3__17complexIdEELNS_13runtime_flagsE0EEENS3_9enable_ifIXoosr10is_complexIT_EE5valuesr10is_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE18EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(double, %"class.std::__1::complex"*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6mod_opIdNSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr10is_complexIT_EE5valuesr10is_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE18EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(double, %"class.std::__1::complex"*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6mod_opIxNSt3__17complexIdEELNS_13runtime_flagsE0EEENS3_9enable_ifIXoosr10is_complexIT_EE5valuesr10is_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE18EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(i64, %"class.std::__1::complex"*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6mod_opIxNSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr10is_complexIT_EE5valuesr10is_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE18EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(i64, %"class.std::__1::complex"*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN8vfun_modIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN8vfun_modIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN8vfun_modIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i32 @ippsDivC_64fc(%"class.std::__1::complex"*, double, double, %"class.std::__1::complex"*, i32)

declare void @vzDiv(i64, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL9divide_opINSt3__17complexIdEEdLNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE9EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(%"class.std::__1::complex"*, double*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL9divide_opINSt3__17complexIdEExLNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE9EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(%"class.std::__1::complex"*, i64)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL9divide_opIdNSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE9EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(double*, %"class.std::__1::complex"*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i32 @ippsDivC_64f(double*, double, double*, i32)

declare void @vdDiv(i64, double*, double*, double*)

declare i64 @_ZN11vfun_divideIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_divideIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_divideIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_divideIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_divideIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_divideIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_divideIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_divideIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_divideIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_divideIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_divideIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_divideIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL9divide_opIxNSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE9EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(i64, %"class.std::__1::complex"*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN11vfun_divideIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN11vfun_divideIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL8times_opINSt3__17complexIdEES5_LNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE26EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(%"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i32 @ippsMulC_64fc(%"class.std::__1::complex"*, double, double, %"class.std::__1::complex"*, i32)

declare void @vzMul(i64, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL8times_opINSt3__17complexIdEEdLNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE26EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(%"class.std::__1::complex"*, double*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL8times_opINSt3__17complexIdEExLNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE26EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(%"class.std::__1::complex"*, i64)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL8times_opIdNSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE26EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(double*, %"class.std::__1::complex"*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i32 @ippsMulC_64f(double*, double, double*, i32)

declare void @vdMul(i64, double*, double*, double*)

declare i64 @_ZN10vfun_timesIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL8times_opIxNSt3__17complexIdEELNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE26EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(i64, %"class.std::__1::complex"*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_timesIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN10vfun_timesIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN10vfun_timesIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i32 @ippsSubC_64fc(%"class.std::__1::complex"*, double, double, %"class.std::__1::complex"*, i32)

declare i32 @ippsSubCRev_64fc(%"class.std::__1::complex"*, double, double, %"class.std::__1::complex"*, i32)

declare void @vzSub(i64, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i32 @ippsSubC_64f(double*, double, double*, i32)

declare i32 @ippsSubCRev_64f(double*, double, double*, i32)

declare void @vdSub(i64, double*, double*, double*)

declare i64 @_ZN13vfun_subtractIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN13vfun_subtractIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i32 @ippsAddC_64fc(%"class.std::__1::complex"*, double, double, %"class.std::__1::complex"*, i32)

declare void @vzAdd(i64, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKS2_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_dE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKS2_PKdRKxPSD_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEEdS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKdPKS2_RKxPSD_(%"class.std::__1::complex"*, double*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusIdddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i32 @ippsAddC_64f(double*, double, double*, i32)

declare void @vdAdd(i64, double*, double*, double*)

declare i64 @_ZN9vfun_plusIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusIdddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusIdddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKdS7_RKxPS8_(double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusIddxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusIddxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusIddxE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKdPKxRS8_S9_(double*, double*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusINSt3__17complexIdEExS2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKxPKS2_RS9_SA_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_plusIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusIdxdE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class2EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusIdxdE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class3EEExPdPKxPKdRS6_S7_(double*, i64*, double*, i64*, i64*)

declare i64 @_ZN9vfun_plusIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_plusIxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPxPKxS7_RS6_S7_(i64*, i64*, i64*, i64*, i64*)

declare i64 @machfunc_bb_bitxor(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZN12vfun_bit_xorIhhhE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPhPKhS7_RKxPS8_(i8*, i8*, i8*, i64*, i64*)

declare i64 @_ZN12vfun_bit_xorIhhhE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPhPKhS7_RKxPS8_(i8*, i8*, i8*, i64*, i64*)

declare void @.omp_outlined..34(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (i8*, i8*, i8*, i64*, i64*)**, i8**, i8**, i8**, i64**)

declare i64 @_ZN12vfun_bit_xorIhhhE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPhPKhS7_RKxPS8_(i8*, i8*, i8*, i64*, i64*)

declare i64 @_ZN12vfun_bit_xorIhhhE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class2EEExPhPKhS7_RKxPS8_(i8*, i8*, i8*, i64*, i64*)

declare i64 @_ZN12vfun_bit_xorIhhhE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class3EEExPhPKhS7_RKxPS8_(i8*, i8*, i8*, i64*, i64*)

declare i64 @_ZN12vfun_bit_xorIhhhE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPhPKhS7_RKxPS8_(i8*, i8*, i8*, i64*, i64*)

declare i64 @_ZN12vfun_bit_xorIhhhE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class2EEExPhPKhS7_RKxPS8_(i8*, i8*, i8*, i64*, i64*)

declare i64 @_ZN12vfun_bit_xorIhhhE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class3EEExPhPKhS7_RKxPS8_(i8*, i8*, i8*, i64*, i64*)

declare i64 @_Z16machfunc_ci_rootPvPKvS1_xPKxj(i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZN9vfun_rootINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_rootINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_rootINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL7root_opINSt3__17complexIdEExLNS_13runtime_flagsE0EEENS3_9enable_ifIXaasr10is_complexIT_EE5valuesr18is_signed_integralIT0_EE5valueENS1_6detail7op_infoILNSA_2opE22EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(%"class.std::__1::complex"*, i64)

declare { double, double } @_ZN3wrt9scalar_op6binaryL7root_opINSt3__17complexIdEExLNS_13runtime_flagsE1EEENS3_9enable_ifIXaasr10is_complexIT_EE5valuesr18is_signed_integralIT0_EE5valueENS1_6detail7op_infoILNSA_2opE22EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(%"class.std::__1::complex"*, i64)

declare i64 @_ZN9vfun_rootINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_rootINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_rootINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_rootINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_rootINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_rootINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_rootINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_rootINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class2EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_rootINSt3__17complexIdEES2_xE7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class3EEExPS2_PKS2_PKxRSB_SC_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*, i64*)

declare i32 @ReturnRangeType2(i32, i32, i32)

declare i32 @ReturnType2(i32, i32, i32)

declare i64 @machfunc_iii_axpy(i8*, i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZN9vfun_axpyIxxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxS7_S7_RS6_S7_(i64*, i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_axpyIxxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxS7_S7_RS6_S7_(i64*, i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_axpyIxxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxS7_S7_RS6_S7_(i64*, i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_axpyIxxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxS7_S7_RS6_S7_(i64*, i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZL9vfun3_runIxxxxExPFxPT_PKT0_PKT1_PKT2_RKxPSB_ES1_S4_S7_SA_SC_SD_(i64 (i64*, i64*, i64*, i64*, i64*, i64*)*, i64*, i64*, i64*, i64*, i64*, i64*)

declare void @.omp_outlined..53(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (i64*, i64*, i64*, i64*, i64*, i64*)**, i64**, i64**, i64**, i64**, i64**)

declare i64 @_ZN3wrt9scalar_op7ternaryL7axpy_opIxxxLNS_13runtime_flagsE1EEENSt3__19enable_ifIXaaaasr11is_integralIT_EE5valuesr11is_integralIT0_EE5valuesr11is_integralIT1_EE5valueENS1_6detail7op_infoILNS9_2opE3EN7rt_typeIS6_E4typeENSC_IS7_E4typeENSC_IS8_E4typeEE11result_typeEE4typeERKNSJ_19first_argument_typeERKNSJ_20second_argument_typeERKNSJ_19third_argument_typeE(i64, i64, i64*)

declare i64 @machfunc_ddd_axpy(i8*, i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZN9vfun_axpyIddddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdS7_S7_RKxPS8_(double*, double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_axpyIddddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdS7_S7_RKxPS8_(double*, double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_axpyIddddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdS7_S7_RKxPS8_(double*, double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_axpyIddddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdS7_S7_RKxPS8_(double*, double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_axpyIddddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdS7_S7_RKxPS8_(double*, double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_axpyIddddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdS7_S7_RKxPS8_(double*, double*, double*, double*, i64*, i64*)

declare i64 @_ZL9vfun3_runIddddExPFxPT_PKT0_PKT1_PKT2_RKxPSB_ES1_S4_S7_SA_SC_SD_(i64 (double*, double*, double*, double*, i64*, i64*)*, double*, double*, double*, double*, i64*, i64*)

declare void @.omp_outlined..2.55(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (double*, double*, double*, double*, i64*, i64*)**, double**, double**, double**, double**, i64**)

declare i64 @machfunc_ccc_axpy(i8*, i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZN9vfun_axpyINSt3__17complexIdEES2_S2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_SA_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_axpyINSt3__17complexIdEES2_S2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_SA_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_axpyINSt3__17complexIdEES2_S2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_SA_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_axpyINSt3__17complexIdEES2_S2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_SA_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_axpyINSt3__17complexIdEES2_S2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_SA_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_axpyINSt3__17complexIdEES2_S2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_SA_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZL9vfun3_runINSt3__17complexIdEES2_S2_S2_ExPFxPT_PKT0_PKT1_PKT2_RKxPSE_ES4_S7_SA_SD_SF_SG_(i64 (%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare void @.omp_outlined..4.56(i32*, i32*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64 (%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)**, %"class.std::__1::complex"**, %"class.std::__1::complex"**, %"class.std::__1::complex"**, %"class.std::__1::complex"**, i64**)

declare { double, double } @_ZN3wrt9scalar_op7ternaryL7axpy_opINSt3__17complexIdEES5_S5_LNS_13runtime_flagsE1EEENS3_9enable_ifIXoooosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valuesr28is_floating_point_or_complexIT1_EE5valueENS1_6detail7op_infoILNSB_2opE3EN7rt_typeIS8_E4typeENSE_IS9_E4typeENSE_ISA_E4typeEE11result_typeEE4typeERKNSL_19first_argument_typeERKNSL_20second_argument_typeERKNSL_19third_argument_typeE(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double, double)

declare i64 @machfunc_iii_axmy(i8*, i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZN9vfun_axmyIxxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxS7_S7_RS6_S7_(i64*, i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_axmyIxxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxS7_S7_RS6_S7_(i64*, i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_axmyIxxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxS7_S7_RS6_S7_(i64*, i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_axmyIxxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxS7_S7_RS6_S7_(i64*, i64*, i64*, i64*, i64*, i64*)

declare i64 @machfunc_ddd_axmy(i8*, i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZN9vfun_axmyIddddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdS7_S7_RKxPS8_(double*, double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_axmyIddddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdS7_S7_RKxPS8_(double*, double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_axmyIddddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdS7_S7_RKxPS8_(double*, double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_axmyIddddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdS7_S7_RKxPS8_(double*, double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_axmyIddddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdS7_S7_RKxPS8_(double*, double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_axmyIddddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdS7_S7_RKxPS8_(double*, double*, double*, double*, i64*, i64*)

declare i64 @machfunc_ccc_axmy(i8*, i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZN9vfun_axmyINSt3__17complexIdEES2_S2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_SA_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_axmyINSt3__17complexIdEES2_S2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_SA_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_axmyINSt3__17complexIdEES2_S2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_SA_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_axmyINSt3__17complexIdEES2_S2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_SA_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_axmyINSt3__17complexIdEES2_S2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_SA_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_axmyINSt3__17complexIdEES2_S2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_SA_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op7ternaryL7axmy_opINSt3__17complexIdEES5_S5_LNS_13runtime_flagsE1EEENS3_9enable_ifIXoooosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valuesr28is_floating_point_or_complexIT1_EE5valueENS1_6detail7op_infoILNSB_2opE2EN7rt_typeIS8_E4typeENSE_IS9_E4typeENSE_ISA_E4typeEE11result_typeEE4typeERKNSL_19first_argument_typeERKNSL_20second_argument_typeERKNSL_19third_argument_typeE(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double, double)

declare i64 @machfunc_iii_ymax(i8*, i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZN9vfun_ymaxIxxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPxPKxS7_S7_RS6_S7_(i64*, i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_ymaxIxxxxE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPxPKxS7_S7_RS6_S7_(i64*, i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_ymaxIxxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPxPKxS7_S7_RS6_S7_(i64*, i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN9vfun_ymaxIxxxxE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPxPKxS7_S7_RS6_S7_(i64*, i64*, i64*, i64*, i64*, i64*)

declare i64 @_ZN3wrt9scalar_op7ternaryL7ymax_opIxxxLNS_13runtime_flagsE1EEENSt3__19enable_ifIXaaaasr11is_integralIT_EE5valuesr11is_integralIT0_EE5valuesr11is_integralIT1_EE5valueENS1_6detail7op_infoILNS9_2opE5EN7rt_typeIS6_E4typeENSC_IS7_E4typeENSC_IS8_E4typeEE11result_typeEE4typeERKNSJ_19first_argument_typeERKNSJ_20second_argument_typeERKNSJ_19third_argument_typeE(i64, i64, i64*)

declare i64 @machfunc_ddd_ymax(i8*, i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZN9vfun_ymaxIddddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdS7_S7_RKxPS8_(double*, double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_ymaxIddddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdS7_S7_RKxPS8_(double*, double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_ymaxIddddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdS7_S7_RKxPS8_(double*, double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_ymaxIddddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdS7_S7_RKxPS8_(double*, double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_ymaxIddddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdS7_S7_RKxPS8_(double*, double*, double*, double*, i64*, i64*)

declare i64 @_ZN9vfun_ymaxIddddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdS7_S7_RKxPS8_(double*, double*, double*, double*, i64*, i64*)

declare i64 @machfunc_ccc_ymax(i8*, i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZN9vfun_ymaxINSt3__17complexIdEES2_S2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_SA_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_ymaxINSt3__17complexIdEES2_S2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_SA_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_ymaxINSt3__17complexIdEES2_S2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_SA_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_ymaxINSt3__17complexIdEES2_S2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_SA_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_ymaxINSt3__17complexIdEES2_S2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_SA_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN9vfun_ymaxINSt3__17complexIdEES2_S2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_SA_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op7ternaryL7ymax_opINSt3__17complexIdEES5_S5_LNS_13runtime_flagsE1EEENS3_9enable_ifIXoooosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valuesr28is_floating_point_or_complexIT1_EE5valueENS1_6detail7op_infoILNSB_2opE5EN7rt_typeIS8_E4typeENSE_IS9_E4typeENSE_ISA_E4typeEE11result_typeEE4typeERKNSL_19first_argument_typeERKNSL_20second_argument_typeERKNSL_19third_argument_typeE(%"class.std::__1::complex"*, %"class.std::__1::complex"*, double, double)

declare i64 @machfunc_ddd_adxmy(i8*, i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZN10vfun_adxmyIddddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class1EEExPdPKdS7_S7_RKxPS8_(double*, double*, double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_adxmyIddddE7iterateILN3wrt13runtime_flagsE1ELS3_0EL15increment_class0EEExPdPKdS7_S7_RKxPS8_(double*, double*, double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_adxmyIddddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class1EEExPdPKdS7_S7_RKxPS8_(double*, double*, double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_adxmyIddddE7iterateILN3wrt13runtime_flagsE0ELS3_1EL15increment_class0EEExPdPKdS7_S7_RKxPS8_(double*, double*, double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_adxmyIddddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class1EEExPdPKdS7_S7_RKxPS8_(double*, double*, double*, double*, i64*, i64*)

declare i64 @_ZN10vfun_adxmyIddddE7iterateILN3wrt13runtime_flagsE0ELS3_0EL15increment_class0EEExPdPKdS7_S7_RKxPS8_(double*, double*, double*, double*, i64*, i64*)

declare double @_ZN3wrt9scalar_op7ternaryL8adxmy_opIdddLNS_13runtime_flagsE1EEENSt3__19enable_ifIXoooosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valuesr28is_floating_point_or_complexIT1_EE5valueENS1_6detail7op_infoILNS9_2opE1EN7rt_typeIS6_E4typeENSC_IS7_E4typeENSC_IS8_E4typeEE11result_typeEE4typeERKNSJ_19first_argument_typeERKNSJ_20second_argument_typeERKNSJ_19third_argument_typeE(double, double, double)

declare i64 @machfunc_ccc_adxmy(i8*, i8*, i8*, i8*, i64, i64*, i32)

declare i64 @_ZN10vfun_adxmyINSt3__17complexIdEES2_S2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class1EEExPS2_PKS2_SA_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_adxmyINSt3__17complexIdEES2_S2_S2_E7iterateILN3wrt13runtime_flagsE1ELS6_0EL15increment_class0EEExPS2_PKS2_SA_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_adxmyINSt3__17complexIdEES2_S2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class1EEExPS2_PKS2_SA_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_adxmyINSt3__17complexIdEES2_S2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_1EL15increment_class0EEExPS2_PKS2_SA_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_adxmyINSt3__17complexIdEES2_S2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class1EEExPS2_PKS2_SA_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare i64 @_ZN10vfun_adxmyINSt3__17complexIdEES2_S2_S2_E7iterateILN3wrt13runtime_flagsE0ELS6_0EL15increment_class0EEExPS2_PKS2_SA_SA_RKxPSB_(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, i64*)

declare { double, double } @_ZN3wrt9scalar_op7ternaryL8adxmy_opINSt3__17complexIdEES5_S5_LNS_13runtime_flagsE1EEENS3_9enable_ifIXoooosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valuesr28is_floating_point_or_complexIT1_EE5valueENS1_6detail7op_infoILNSB_2opE1EN7rt_typeIS8_E4typeENSE_IS9_E4typeENSE_ISA_E4typeEE11result_typeEE4typeERKNSL_19first_argument_typeERKNSL_20second_argument_typeERKNSL_19third_argument_typeE(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare i64 @machfunc_ddd_compplus(double*, double*, double*, double*, i64, i64*, i32)

declare i64 @machfunc_ccc_compplus(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64, i64*, i32)

declare i32 @machfunc_d_vectornorm(double*, i64, double*, i64, i64, i32)

declare double @dasum_(i64*, double*, i64*)

declare double @dnrm2_(i64*, double*, i64*)

declare i64 @idamax_(i64*, double*, i64*)

declare i32 @machfunc_c_vectornorm(double*, i64, %"class.std::__1::complex"*, i64, i64, i32)

declare double @dznrm2_(i64*, %"class.std::__1::complex"*, i64*)

declare i32 @machfunc_d_weightedvectornorm(double*, i64, double*, double*, i64, i32, i32)

declare i32 @machfunc_c_weightedvectornorm(double*, i64, %"class.std::__1::complex"*, double*, i64, i32, i32)

declare i64 @machfunc_d_vectornormweights(double*, double, double, double*, i64, i32)

declare i64 @machfunc_c_vectornormweights(double*, double, double, %"class.std::__1::complex"*, i64, i32)

declare i32 @machfunc_d_scaledvectornorm(double*, i64, double, double, double*, double*, i64, i32, i32)

declare i32 @machfunc_c_scaledvectornorm(double*, i64, double, double, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64, i32, i32)

declare i32 @getProcessorCount()

declare i32 @getHyperThreadCount()

declare i32 @getMKLThreadNumber()

declare i32 @setMKLThreadNumber(i32)

declare i32 @MKL_Domain_Set_Num_Threads(i32, i32)

declare void @restoreMKLThreadNumber(i32)

declare i32 @getParallelThreadNumber()

declare i32 @setParallelThreadNumber(i32)

declare void @omp_set_num_threads(i32)

declare void @restoreParallelThreadNumber(i32)

declare void @_Z20init_parallel_paramsi(i32)

declare void @MKL_Set_Dynamic(i32)

declare i32 @MKL_Get_Max_Threads()

declare i32 @vmlSetMode(i64)

declare i32 @ParallelThreadsEnvironment_initializedQ()

declare %struct.st_ParallelThreadsSchedule* @New_ParallelThreadsSchedule(i64, i64)

declare %struct.st_ParallelThreadsSchedule* @New_ParallelThreadsScheduleSingle()

declare void @Delete_ParallelThreadsSchedule(%struct.st_ParallelThreadsSchedule*)

declare void @InitializeParallelThreads()

declare %struct.st_ParallelThreadsEnvironment* @_ZL30New_ParallelThreadsEnvironmentx(i64)

declare void @_ZL17runParallelThreadPv(i8*)

declare void @DeinitializeParallelThreads()

declare void @ParallelThreads_addStartFunction(void (%struct.st_ParallelThread*)*)

declare i32 @ParallelThreads_For(i8*, void (i8*, %struct.st_ParallelThreadsEnvironment*)*, void (i8*, %struct.st_ParallelThread*)*, i64, i64, i64, %struct.st_ParallelThreadsSchedule*)

declare i32 @ParallelThreads_Reduce(i8*, void (i8*, %struct.st_ParallelThreadsEnvironment*)*, void (i8*, %struct.st_ParallelThread*)*, i64, i64, i64)

declare i32 @ParallelThreads_Iterate(i8*, void (i8*, %struct.st_ParallelThread*)*, i32)

declare i64 @ParallelThread_RangeStart(%struct.st_ParallelThread*)

declare i64 @ParallelThread_RangeEnd(%struct.st_ParallelThread*)

declare i64 @ParallelThread_RangeStep(%struct.st_ParallelThread*)

declare i64 @ParallelThread_ID(%struct.st_ParallelThread*)

declare void @ParallelThread_synchronizeLock(%struct.st_ParallelThread*)

declare void @ParallelThread_synchronizeUnlock(%struct.st_ParallelThread*)

declare void @ParallelThread_cancel(%struct.st_ParallelThread*, i32)

declare i32 @ParallelThread_canceledQ(%struct.st_ParallelThread*)

declare i64 @ParallelThreadsEnvironment_numThreads(%struct.st_ParallelThreadsEnvironment*)

declare i64* @ParallelThreadID(i64*)

declare void @_Z21init_parallel_threadsi(i32)

declare i32 @s_machfunc_i_abs(i64*, i64, i32)

declare i32 @s_machfunc_d_abs(double*, double, i32)

declare i32 @s_machfunc_c_abs(double*, double, double, i32)

declare i32 @s_machfunc_i_conj(i64*, i64, i32)

declare i32 @s_machfunc_d_conj(double*, double, i32)

declare i32 @s_machfunc_c_conj(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_c_arg(double*, double, double, i32)

declare i32 @s_machfunc_i_abssquare(i64*, i64, i32)

declare i32 @s_machfunc_d_abssquare(double*, double, i32)

declare i32 @s_machfunc_c_abssquare(double*, double, double, i32)

declare i32 @s_machfunc_d_exp(double*, double, i32)

declare i32 @s_machfunc_c_exp(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_expm1(double*, double, i32)

declare i32 @s_machfunc_c_expm1(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_log(double*, double, i32)

declare i32 @s_machfunc_c_log(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_log1p(double*, double, i32)

declare i32 @s_machfunc_c_log1p(%"class.std::__1::complex"*, double, double, i32)

declare { double, double } @_ZN3wrt9scalar_op5unaryL8log1p_opINSt3__17complexIdEELNS_13runtime_flagsE0EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE47EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE.65(double, double)

declare i32 @s_machfunc_i_log2(i64*, i64, i32)

declare i32 @s_machfunc_d_log2(double*, double, i32)

declare i32 @s_machfunc_c_log2(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_i_log10(i64*, i64, i32)

declare i32 @s_machfunc_d_log10(double*, double, i32)

declare i32 @s_machfunc_c_log10(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_i_zero(i64*, i64, i32)

declare i32 @s_machfunc_d_zero(double*, double, i32)

declare i32 @s_machfunc_c_zero(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_i_copy(i64*, i64, i32)

declare i32 @s_machfunc_d_copy(double*, double, i32)

declare i32 @s_machfunc_c_copy(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_i_minus(i64*, i64, i32)

declare i32 @s_machfunc_d_minus(double*, double, i32)

declare i32 @s_machfunc_c_minus(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_i_re(i64*, i64, i32)

declare i32 @s_machfunc_d_re(double*, double, i32)

declare i32 @s_machfunc_c_re(double*, double, double, i32)

declare i32 @s_machfunc_i_im(i64*, i64, i32)

declare i32 @s_machfunc_d_im(i64*, double, i32)

declare i32 @s_machfunc_c_im(double*, double, double, i32)

declare i32 @s_machfunc_d_recip(double*, double, i32)

declare i32 @s_machfunc_c_recip(%"class.std::__1::complex"*, double, double, i32)

declare { double, double } @_ZN3wrt9scalar_op6binaryL9divide_opINSt3__17complexIdEES5_LNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE9EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE.66(%"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare i32 @s_machfunc_d_sqrt(double*, double, i32)

declare i32 @s_machfunc_c_sqrt(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_cbrt(double*, double, i32)

declare i32 @s_machfunc_d_rsqrt(double*, double, i32)

declare i32 @s_machfunc_c_rsqrt(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_i_square(i64*, i64, i32)

declare i32 @s_machfunc_d_square(double*, double, i32)

declare i32 @s_machfunc_c_square(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_i_round(i64*, i64, i32)

declare i32 @s_machfunc_d_round(i64*, double, i32)

declare i32 @s_machfunc_i_floor(i64*, i64, i32)

declare i32 @s_machfunc_d_floor(i64*, double, i32)

declare i32 @s_machfunc_i_ceiling(i64*, i64, i32)

declare i32 @s_machfunc_d_ceiling(i64*, double, i32)

declare i32 @s_machfunc_i_mod1(i64*, i64, i32)

declare i32 @s_machfunc_d_mod1(double*, double, i32)

declare i32 @s_machfunc_c_mod1(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_i_fracpart(i64*, i64, i32)

declare i32 @s_machfunc_d_fracpart(double*, double, i32)

declare i32 @s_machfunc_c_fracpart(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_i_intpart(i64*, i64, i32)

declare i32 @s_machfunc_d_intpart(i64*, double, i32)

declare i32 @s_machfunc_i_sign(i64*, i64, i32)

declare i32 @s_machfunc_d_sign(i64*, double, i32)

declare i32 @s_machfunc_c_sign(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_sin(double*, double, i32)

declare i32 @s_machfunc_c_sin(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_sinh(double*, double, i32)

declare i32 @s_machfunc_c_sinh(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_cos(double*, double, i32)

declare i32 @s_machfunc_c_cos(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_cosh(double*, double, i32)

declare i32 @s_machfunc_c_cosh(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_tan(double*, double, i32)

declare i32 @s_machfunc_c_tan(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_tanh(double*, double, i32)

declare i32 @s_machfunc_c_tanh(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_csc(double*, double, i32)

declare i32 @s_machfunc_c_csc(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_csch(double*, double, i32)

declare i32 @s_machfunc_c_csch(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_sec(double*, double, i32)

declare i32 @s_machfunc_c_sec(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_sech(double*, double, i32)

declare i32 @s_machfunc_c_sech(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_cot(double*, double, i32)

declare i32 @s_machfunc_c_cot(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_coth(double*, double, i32)

declare i32 @s_machfunc_c_coth(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_asin(double*, double, i32)

declare i32 @s_machfunc_c_asin(%"class.std::__1::complex"*, double, double, i32)

declare { double, double } @_ZN3wrt9scalar_op5unaryL8asinh_opINSt3__17complexIdEELNS_13runtime_flagsE0EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE13EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE.68(%"class.std::__1::complex"*)

declare i32 @s_machfunc_d_asinh(double*, double, i32)

declare i32 @s_machfunc_c_asinh(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_acos(double*, double, i32)

declare i32 @s_machfunc_c_acos(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_acosh(double*, double, i32)

declare i32 @s_machfunc_c_acosh(%"class.std::__1::complex"*, double, double, i32)

declare { double, double } @_ZN3wrt9scalar_op5unaryL8acosh_opINSt3__17complexIdEELNS_13runtime_flagsE0EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE4EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE.69(%"class.std::__1::complex"*)

declare i32 @s_machfunc_d_atan(double*, double, i32)

declare i32 @s_machfunc_c_atan(%"class.std::__1::complex"*, double, double, i32)

declare { double, double } @_ZN3wrt9scalar_op5unaryL8atanh_opINSt3__17complexIdEELNS_13runtime_flagsE0EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE15EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE.70(%"class.std::__1::complex"*)

declare i32 @s_machfunc_d_atanh(double*, double, i32)

declare i32 @s_machfunc_c_atanh(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_acsc(double*, double, i32)

declare i32 @s_machfunc_c_acsc(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_acsch(double*, double, i32)

declare i32 @s_machfunc_c_acsch(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_asec(double*, double, i32)

declare i32 @s_machfunc_c_asec(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_asech(double*, double, i32)

declare i32 @s_machfunc_c_asech(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_acot(double*, double, i32)

declare i32 @s_machfunc_c_acot(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_acoth(double*, double, i32)

declare i32 @s_machfunc_c_acoth(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_i_pos(i64*, i64, i32)

declare i32 @s_machfunc_d_pos(i64*, double, i32)

declare i32 @s_machfunc_i_neg(i64*, i64, i32)

declare i32 @s_machfunc_d_neg(i64*, double, i32)

declare i32 @s_machfunc_i_npos(i64*, i64, i32)

declare i32 @s_machfunc_d_npos(i64*, double, i32)

declare i32 @s_machfunc_i_nneg(i64*, i64, i32)

declare i32 @s_machfunc_d_nneg(i64*, double, i32)

declare i32 @s_machfunc_i_evenq(i64*, i64, i32)

declare i32 @s_machfunc_i_oddq(i64*, i64, i32)

declare i32 @s_machfunc_i_bitlength(i64*, i64, i32)

declare i32 @s_machfunc_i_bitnot(i64*, i64, i32)

declare i32 @s_machfunc_i_ramp(i64*, i64, i32)

declare i32 @s_machfunc_d_ramp(double*, double, i32)

declare i32 @s_machfunc_i_unitize(i64*, i64, i32)

declare i32 @s_machfunc_d_unitize(i64*, double, i32)

declare i32 @s_machfunc_c_unitize(i64*, double, double, i32)

declare i32 @s_machfunc_d_sinc(double*, double, i32)

declare i32 @s_machfunc_c_sinc(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_i_fibonacci(i64*, i64, i32)

declare i32 @s_machfunc_d_fibonacci(double*, double, i32)

declare i32 @s_machfunc_c_fibonacci(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_i_lucasl(i64*, i64, i32)

declare i32 @s_machfunc_d_lucasl(double*, double, i32)

declare i32 @s_machfunc_c_lucasl(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_erf(double*, double, i32)

declare i32 @s_machfunc_d_erfc(double*, double, i32)

declare i32 @s_machfunc_i_gamma(i64*, i64, i32)

declare i32 @s_machfunc_d_gamma(double*, double, i32)

declare i32 @s_machfunc_d_loggamma(double*, double, i32)

declare i32 @s_machfunc_d_gudermannian(double*, double, i32)

declare i32 @s_machfunc_c_gudermannian(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_inversegudermannian(double*, double, i32)

declare i32 @s_machfunc_c_inversegudermannian(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_haversine(double*, double, i32)

declare i32 @s_machfunc_c_haversine(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_inversehaversine(double*, double, i32)

declare i32 @s_machfunc_c_inversehaversine(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_d_logistic(double*, double, i32)

declare i32 @s_machfunc_c_logistic(%"class.std::__1::complex"*, double, double, i32)

declare i32 @s_machfunc_ii_plus(i64*, i64, i64, i32)

declare i32 @s_machfunc_id_plus(double*, i64, double, i32)

declare i32 @s_machfunc_di_plus(double*, double, i64, i32)

declare i32 @s_machfunc_dd_plus(double*, double, double, i32)

declare i32 @s_machfunc_ic_plus(%"class.std::__1::complex"*, i64, double, double, i32)

declare i32 @s_machfunc_ci_plus(%"class.std::__1::complex"*, double, double, i64, i32)

declare i32 @s_machfunc_dc_plus(%"class.std::__1::complex"*, double, double, double, i32)

declare i32 @s_machfunc_cd_plus(%"class.std::__1::complex"*, double, double, double, i32)

declare i32 @s_machfunc_cc_plus(%"class.std::__1::complex"*, double, double, double, double, i32)

declare i32 @s_machfunc_ii_subtract(i64*, i64, i64, i32)

declare i32 @s_machfunc_id_subtract(double*, i64, double, i32)

declare i32 @s_machfunc_di_subtract(double*, double, i64, i32)

declare i32 @s_machfunc_dd_subtract(double*, double, double, i32)

declare i32 @s_machfunc_ic_subtract(%"class.std::__1::complex"*, i64, double, double, i32)

declare i32 @s_machfunc_ci_subtract(%"class.std::__1::complex"*, double, double, i64, i32)

declare i32 @s_machfunc_dc_subtract(%"class.std::__1::complex"*, double, double, double, i32)

declare i32 @s_machfunc_cd_subtract(%"class.std::__1::complex"*, double, double, double, i32)

declare i32 @s_machfunc_cc_subtract(%"class.std::__1::complex"*, double, double, double, double, i32)

declare i32 @s_machfunc_ii_times(i64*, i64, i64, i32)

declare i32 @s_machfunc_id_times(double*, i64, double, i32)

declare i32 @s_machfunc_di_times(double*, double, i64, i32)

declare i32 @s_machfunc_dd_times(double*, double, double, i32)

declare i32 @s_machfunc_ic_times(%"class.std::__1::complex"*, i64, double, double, i32)

declare i32 @s_machfunc_ci_times(%"class.std::__1::complex"*, double, double, i64, i32)

declare i32 @s_machfunc_dc_times(%"class.std::__1::complex"*, double, double, double, i32)

declare i32 @s_machfunc_cd_times(%"class.std::__1::complex"*, double, double, double, i32)

declare i32 @s_machfunc_cc_times(%"class.std::__1::complex"*, double, double, double, double, i32)

declare i32 @s_machfunc_id_div(double*, i64, double, i32)

declare i32 @s_machfunc_di_div(double*, double, i64, i32)

declare i32 @s_machfunc_dd_div(double*, double, double, i32)

declare i32 @s_machfunc_ic_div(%"class.std::__1::complex"*, i64, double, double, i32)

declare i32 @s_machfunc_ci_div(%"class.std::__1::complex"*, double, double, i64, i32)

declare i32 @s_machfunc_dc_div(%"class.std::__1::complex"*, double, double, double, i32)

declare i32 @s_machfunc_cd_div(%"class.std::__1::complex"*, double, double, double, i32)

declare i32 @s_machfunc_cc_div(%"class.std::__1::complex"*, double, double, double, double, i32)

declare { double, double } @_ZN3wrt9scalar_op6binaryL9divide_opINSt3__17complexIdEES5_LNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE9EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE.71(%"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare i32 @s_machfunc_ii_abserr(i64*, i64, i64, i32)

declare i32 @s_machfunc_id_abserr(double*, i64, double, i32)

declare i32 @s_machfunc_di_abserr(double*, double, i64, i32)

declare i32 @s_machfunc_dd_abserr(double*, double, double, i32)

declare i32 @s_machfunc_ic_abserr(double*, i64, double, double, i32)

declare i32 @s_machfunc_ci_abserr(double*, double, double, i64, i32)

declare i32 @s_machfunc_dc_abserr(double*, double, double, double, i32)

declare i32 @s_machfunc_cd_abserr(double*, double, double, double, i32)

declare i32 @s_machfunc_cc_abserr(double*, double, double, double, double, i32)

declare i32 @s_machfunc_ii_maxabs(i64*, i64, i64, i32)

declare i32 @s_machfunc_dd_maxabs(double*, double, double, i32)

declare i32 @s_machfunc_cc_maxabs(double*, double, double, double, double, i32)

declare i32 @s_machfunc_id_relerr(double*, i64, double, i32)

declare i32 @s_machfunc_di_relerr(double*, double, i64, i32)

declare i32 @s_machfunc_dd_relerr(double*, double, double, i32)

declare i32 @s_machfunc_ic_relerr(double*, i64, double, double, i32)

declare i32 @s_machfunc_ci_relerr(double*, double, double, i64, i32)

declare i32 @s_machfunc_dc_relerr(double*, double, double, double, i32)

declare i32 @s_machfunc_cd_relerr(double*, double, double, double, i32)

declare i32 @s_machfunc_cc_relerr(double*, double, double, double, double, i32)

declare i32 @s_machfunc_id_log(double*, i64, double, i32)

declare i32 @s_machfunc_di_log(double*, double, i64, i32)

declare i32 @s_machfunc_dd_log(double*, double, double, i32)

declare i32 @s_machfunc_ic_log(%"class.std::__1::complex"*, i64, double, double, i32)

declare i32 @s_machfunc_ci_log(%"class.std::__1::complex"*, double, double, i64, i32)

declare i32 @s_machfunc_dc_log(%"class.std::__1::complex"*, double, double, double, i32)

declare i32 @s_machfunc_cd_log(%"class.std::__1::complex"*, double, double, double, i32)

declare i32 @s_machfunc_cc_log(%"class.std::__1::complex"*, double, double, double, double, i32)

declare i32 @s_machfunc_id_atan2(double*, i64, double, i32)

declare i32 @s_machfunc_di_atan2(double*, double, i64, i32)

declare i32 @s_machfunc_dd_atan2(double*, double, double, i32)

declare i32 @s_machfunc_ic_atan2(%"class.std::__1::complex"*, i64, double, double, i32)

declare i32 @s_machfunc_ci_atan2(%"class.std::__1::complex"*, double, double, i64, i32)

declare i32 @s_machfunc_dc_atan2(%"class.std::__1::complex"*, double, double, double, i32)

declare i32 @s_machfunc_cd_atan2(%"class.std::__1::complex"*, double, double, double, i32)

declare i32 @s_machfunc_cc_atan2(%"class.std::__1::complex"*, double, double, double, double, i32)

declare i32 @s_machfunc_ii_mod(i64*, i64, i64, i32)

declare i32 @s_machfunc_id_mod(double*, i64, double, i32)

declare i32 @s_machfunc_di_mod(double*, double, i64, i32)

declare i32 @s_machfunc_dd_mod(double*, double, double, i32)

declare i32 @s_machfunc_ic_mod(%"class.std::__1::complex"*, i64, double, double, i32)

declare i32 @s_machfunc_ci_mod(%"class.std::__1::complex"*, double, double, i64, i32)

declare i32 @s_machfunc_dc_mod(%"class.std::__1::complex"*, double, double, double, i32)

declare i32 @s_machfunc_cd_mod(%"class.std::__1::complex"*, double, double, double, i32)

declare i32 @s_machfunc_cc_mod(%"class.std::__1::complex"*, double, double, double, double, i32)

declare i32 @s_machfunc_ii_pow(i64*, i64, i64, i32)

declare i64 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIxxLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_.73(i64, i64)

declare i32 @s_machfunc_di_pow(double*, double, i64, i32)

declare double @_ZN3wrt9scalar_op6binaryL6pow_opIdxLNS_13runtime_flagsE1EEENSt3__19enable_ifIXaasr17is_floating_pointIT_EE5valuesr11is_integralIT0_EE5valueENS1_6detail7op_infoILNS8_2opE20EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE.76(double*, i64)

declare double @_ZN3wrt9scalar_op6binaryL9pow_xi_opIdxLNS_13runtime_flagsE0EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_.77(double, i64)

declare i32 @s_machfunc_ci_pow(%"class.std::__1::complex"*, double, double, i64, i32)

declare { double, double } @_ZN3wrt9scalar_op6binaryL9pow_xi_opINSt3__17complexIdEExLNS_13runtime_flagsE0EEENS3_9enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS9_RKS8_.78(%"class.std::__1::complex"*, i64)

declare i32 @s_machfunc_id_pow(double*, i64, double, i32)

declare i32 @s_machfunc_dd_pow(double*, double, double, i32)

declare i32 @s_machfunc_cd_pow(%"class.std::__1::complex"*, double, double, double, i32)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6pow_opINSt3__17complexIdEEdLNS_13runtime_flagsE1EEENS3_9enable_ifIXaasr10is_complexIT_EE5valuesr17is_floating_pointIT0_EE5valueENS1_6detail7op_infoILNSA_2opE20EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE.79(double, double, double*)

declare { double, double } @_ZN3wrt9scalar_op6binaryL6pow_opINSt3__17complexIdEEdLNS_13runtime_flagsE0EEENS3_9enable_ifIXaasr10is_complexIT_EE5valuesr17is_floating_pointIT0_EE5valueENS1_6detail7op_infoILNSA_2opE20EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE.80(double, double, double*)

declare i32 @s_machfunc_ic_pow(%"class.std::__1::complex"*, i64, double, double, i32)

declare i32 @s_machfunc_dc_pow(%"class.std::__1::complex"*, double, double, double, i32)

declare i32 @s_machfunc_cc_pow(%"class.std::__1::complex"*, double, double, double, double, i32)

declare i32 @s_machfunc_ii_quotient(i64*, i64, i64, i32)

declare i32 @s_machfunc_di_quotient(i64*, double, i64, i32)

declare i32 @s_machfunc_id_quotient(i64*, i64, double, i32)

declare i32 @s_machfunc_dd_quotient(i64*, double, double, i32)

declare i32 @s_machfunc_ci_root(%"class.std::__1::complex"*, double, double, i64, i32)

declare i32 @s_machfunc_ii_bitand(i64*, i64, i64, i32)

declare i32 @s_machfunc_ii_bitor(i64*, i64, i64, i32)

declare i32 @s_machfunc_ii_bitxor(i64*, i64, i64, i32)

declare i32 @s_machfunc_ii_bitshiftleft(i64*, i64, i64, i32)

declare i32 @s_machfunc_di_bitshiftleft(double*, double, i64, i32)

declare i32 @s_machfunc_ci_bitshiftleft(%"class.std::__1::complex"*, double, double, i64, i32)

declare i32 @s_machfunc_ii_bitshiftright(i64*, i64, i64, i32)

declare i32 @s_machfunc_di_bitshiftright(double*, double, i64, i32)

declare i32 @s_machfunc_ci_bitshiftright(%"class.std::__1::complex"*, double, double, i64, i32)

declare i32 @s_machfunc_ii_intexp(i64*, i64, i64, i32)

declare i32 @s_machfunc_ii_intlen(i64*, i64, i64, i32)

declare i32 @s_machfunc_dd_chop(double*, double, double, i32)

declare i32 @s_machfunc_cd_chop(%"class.std::__1::complex"*, double, double, double, i32)

declare i32 @s_machfunc_ii_unitize(i64*, i64, i64, i32)

declare i32 @s_machfunc_dd_unitize(i64*, double, double, i32)

declare i32 @s_machfunc_cd_unitize(i64*, double, double, double, i32)

declare i32 @s_machfunc_iii_axpy(i64*, i64, i64, i64, i32)

declare i32 @s_machfunc_ddd_axpy(double*, double, double, double, i32)

declare i32 @s_machfunc_ccc_axpy(%"class.std::__1::complex"*, double, double, double, double, double, double, i32)

declare i32 @s_machfunc_iii_axmy(i64*, i64, i64, i64, i32)

declare i32 @s_machfunc_ddd_axmy(double*, double, double, double, i32)

declare i32 @s_machfunc_ccc_axmy(%"class.std::__1::complex"*, double, double, double, double, double, double, i32)

declare i32 @s_machfunc_iii_ymax(i64*, i64, i64, i64, i32)

declare i32 @s_machfunc_ddd_ymax(double*, double, double, double, i32)

declare i32 @s_machfunc_ccc_ymax(%"class.std::__1::complex"*, double, double, double, double, double, double, i32)

declare i32 @s_machfunc_ddd_adxmy(double*, double, double, double, i32)

declare i32 @s_machfunc_ccc_adxmy(%"class.std::__1::complex"*, double, double, double, double, double, double, i32)

declare i32 @s_machfunc_ddd_compplus(double*, double, double, double*, i32)

declare i32 @s_machfunc_ccc_compplus(%"class.std::__1::complex"*, double, double, double, double, %"class.std::__1::complex"*, i32)

declare i32 @_ZN3wrt11fpexceptionILNS_13runtime_flagsE1EE8classifyINSt3__14pairINS4_7complexIdEES7_EEEENS4_9enable_ifIXsr7is_pairIT_EE5valueE5errorE4typeERKSA_(%"struct.std::__1::pair.0"*)

declare i64 @getVectorParallelLengthThreshold(i64)

declare void @setVectorParallelLengthThreshold(i64, i64)

declare void @_Z27init_vector_parallel_paramsi(i32)

declare void @memset_pattern16(i8*, i8*, i64)

declare i64 @getVectorVendorLengthThreshold(i64)

declare void @setVectorVendorLengthThreshold(i64, i64)

declare void @_Z25init_vector_vendor_paramsi(i32)

declare i32 @MTensor_getsetMTensor(%struct.st_MDataArray**, %struct.st_MDataArray*, i64*, i64, i32)

declare i32 @memcmp(i8*, i8*, i64)

declare i32 @initializeWolframDLLFunctions(%struct.st_WolframLibraryData*, i64)

declare void @_ZL17UTF8String_disownPc(i8*)

declare i32 @_ZL22MTensor_newFunDLL_LP64iiPKiPP13st_MDataArray(i32, i32, i32*, %struct.st_MDataArray**)

declare void @_ZL12MTensor_freeP13st_MDataArray(%struct.st_MDataArray*)

declare i32 @_ZL13MTensor_cloneP13st_MDataArrayPS0_(%struct.st_MDataArray*, %struct.st_MDataArray**)

declare i64 @_ZL18MTensor_shareCountP13st_MDataArray(%struct.st_MDataArray*)

declare void @_ZL14MTensor_disownP13st_MDataArray(%struct.st_MDataArray*)

declare void @_ZL17MTensor_disownAllP13st_MDataArray(%struct.st_MDataArray*)

declare i32 @_ZL23MTensor_setInteger_LP64P13st_MDataArrayPii(%struct.st_MDataArray*, i32*, i32)

declare i32 @_ZL20MTensor_setReal_LP64P13st_MDataArrayPid(%struct.st_MDataArray*, i32*, double)

declare i32 @_ZL23MTensor_setComplex_LP64P13st_MDataArrayPiNSt3__17complexIdEE(%struct.st_MDataArray*, i32*, double, double)

declare i32 @_ZL23MTensor_setMTensor_LP64P13st_MDataArrayS0_Pii(%struct.st_MDataArray*, %struct.st_MDataArray*, i32*, i32)

declare i32 @_ZL23MTensor_getInteger_LP64P13st_MDataArrayPiS1_(%struct.st_MDataArray*, i32*, i32*)

declare i32 @_ZL20MTensor_getReal_LP64P13st_MDataArrayPiPd(%struct.st_MDataArray*, i32*, double*)

declare i32 @_ZL23MTensor_getComplex_LP64P13st_MDataArrayPiPNSt3__17complexIdEE(%struct.st_MDataArray*, i32*, %"class.std::__1::complex"*)

declare i32 @_ZL23MTensor_getMTensor_LP64P13st_MDataArrayPiiPS0_(%struct.st_MDataArray*, i32*, i32, %struct.st_MDataArray**)

declare i32 @_ZL20MTensor_GetRank_LP64P13st_MDataArray(%struct.st_MDataArray*)

declare i32* @_ZL26MTensor_GetDimensions_LP64P13st_MDataArray(%struct.st_MDataArray*)

declare i32 @_ZL20MTensor_GetType_LP64P13st_MDataArray(%struct.st_MDataArray*)

declare i32 @_ZL32MTensor_GetNumberOfElements_LP64P13st_MDataArray(%struct.st_MDataArray*)

declare i32* @_ZL27MTensor_GetIntegerData_LP64P13st_MDataArray(%struct.st_MDataArray*)

declare double* @_ZL19MTensor_GetRealDataP13st_MDataArray(%struct.st_MDataArray*)

declare %"class.std::__1::complex"* @_ZL22MTensor_GetComplexDataP13st_MDataArray(%struct.st_MDataArray*)

declare i64 @_ZL6AbortQv()

declare i32 @_ZL23EvaluateExpression_LP64P21st_WolframLibraryDataPciiPv(%struct.st_WolframLibraryData*, i8*, i32, i32, i8*)

declare i32 @_ZL17MTensor_newFunDLLxxPKxPP13st_MDataArray(i64, i64, i64*, %struct.st_MDataArray**)

declare i32 @_ZL18MTensor_setIntegerP13st_MDataArrayPxx(%struct.st_MDataArray*, i64*, i64)

declare i32 @_ZL15MTensor_setRealP13st_MDataArrayPxd(%struct.st_MDataArray*, i64*, double)

declare i32 @_ZL18MTensor_setComplexP13st_MDataArrayPxNSt3__17complexIdEE(%struct.st_MDataArray*, i64*, double, double)

declare i32 @_ZL18MTensor_setMTensorP13st_MDataArrayS0_Pxx(%struct.st_MDataArray*, %struct.st_MDataArray*, i64*, i64)

declare i32 @_ZL18MTensor_getIntegerP13st_MDataArrayPxS1_(%struct.st_MDataArray*, i64*, i64*)

declare i32 @_ZL15MTensor_getRealP13st_MDataArrayPxPd(%struct.st_MDataArray*, i64*, double*)

declare i32 @_ZL18MTensor_getComplexP13st_MDataArrayPxPNSt3__17complexIdEE(%struct.st_MDataArray*, i64*, %"class.std::__1::complex"*)

declare i32 @_ZL18MTensor_getMTensorP13st_MDataArrayPxxPS0_(%struct.st_MDataArray*, i64*, i64, %struct.st_MDataArray**)

declare i64 @_ZL15MTensor_GetRankP13st_MDataArray(%struct.st_MDataArray*)

declare i64* @_ZL21MTensor_GetDimensionsP13st_MDataArray(%struct.st_MDataArray*)

declare i64 @_ZL15MTensor_GetTypeP13st_MDataArray(%struct.st_MDataArray*)

declare i64 @_ZL27MTensor_GetNumberOfElementsP13st_MDataArray(%struct.st_MDataArray*)

declare i64* @_ZL22MTensor_GetIntegerDataP13st_MDataArray(%struct.st_MDataArray*)

declare %struct.st_WolframCompileLibrary_Functions* @getWolframCompileDLLFunctions(i64)

declare %struct.M_TENSOR_INITIALIZATION_DATA_STRUCT* @_ZL27GetInitializedMTensors_LP64P21st_WolframLibraryDatai(%struct.st_WolframLibraryData*, i32)

declare void @_ZL29WF_WolframLibraryData_cleanUpP21st_WolframLibraryDatai(%struct.st_WolframLibraryData*, i32)

declare i32 @_ZL24WF_MTensor_allocate_LP64PP13st_MDataArrayiiPi(%struct.st_MDataArray**, i32, i32, i32*)

declare i32 @_ZL15WF_MTensor_copyPP13st_MDataArrayS0_(%struct.st_MDataArray**, %struct.st_MDataArray*)

declare i32 @_ZL26WF_MTensor_copyUnique_LP64PP13st_MDataArrayS0_(%struct.st_MDataArray**, %struct.st_MDataArray*)

declare i32 @_ZL32InsertTensorToCompiledTable_LP64P13st_MDataArrayS0_Pi(%struct.st_MDataArray*, %struct.st_MDataArray*, i32*)

declare i32 @_ZL34MTensor_getMTensorInitialized_LP64PP13st_MDataArrayS0_Pii(%struct.st_MDataArray**, %struct.st_MDataArray*, i32*, i32)

declare i32 @_ZL23WF_MTensor_getPart_LP64PP13st_MDataArrayS0_iPiPPv(%struct.st_MDataArray**, %struct.st_MDataArray*, i32, i32*, i8**)

declare i32 @_ZL23WF_MTensor_setPart_LP64PP13st_MDataArrayS0_iPiPPv(%struct.st_MDataArray**, %struct.st_MDataArray*, i32, i32*, i8**)

declare i64 (i8*, i8*, i64, i64*, i32)* @_ZL28WF_getUnaryMathFunction_LP64ii(i32, i32)

declare i32 @_ZL13Math_T_T_LP64ijP13st_MDataArrayiPS0_(i32, i32, %struct.st_MDataArray*, i32, %struct.st_MDataArray**)

declare i32 @_ZL13Math_V_V_LP64ijiPviS_(i32, i32, i32, i8*, i32, i8*)

declare i64 (i8*, i8*, i8*, i64, i64*, i32)* @_ZL29WF_getBinaryMathFunction_LP64iii(i32, i32, i32)

declare i32 @_ZL14Math_TT_T_LP64ijP13st_MDataArrayS0_iPS0_(i32, i32, %struct.st_MDataArray*, %struct.st_MDataArray*, i32, %struct.st_MDataArray**)

declare i32 @_ZL14Math_VV_V_LP64ijiPviS_iS_(i32, i32, i32, i8*, i32, i8*, i32, i8*)

declare i32 @_ZL17CompareReals_LP64idiPd(i32, double, i32, double*)

declare i32 @_ZL21CompareComplexes_LP64idiPNSt3__17complexIdEE(i32, double, i32, %"class.std::__1::complex"*)

declare i32 @_ZL31evaluateFunctionExpression_LP64P21st_WolframLibraryDataPviiiPiPS1_iiS1_(%struct.st_WolframLibraryData*, i8*, i32, i32, i32, i32*, i8**, i32, i32, i8*)

declare i8** @_ZL32getDLLFunctionArgumentSpace_LP64P21st_WolframLibraryDatai(%struct.st_WolframLibraryData*, i32)

declare i32 (%struct.st_WolframLibraryData*, i64, %union.MArgument*, i32*)* @_ZL26GetDLLFunctionPointer_LP64PcS_(i8*, i8*)

declare i32 (%struct.st_WolframLibraryData*, i64, %union.MArgument*, i32*)* @_ZL28GetFunctionCallFunction_LP64PKc(i8*)

declare %struct.st_MDataArray* @_ZL24LoadRankZeroMTensor_LP64Pvii(i8*, i32, i32)

declare i32 @_ZL19WF_MTensor_allocatePP13st_MDataArrayixPx(%struct.st_MDataArray**, i32, i64, i64*)

declare i32 @_ZL21WF_MTensor_copyUniquePP13st_MDataArrayS0_(%struct.st_MDataArray**, %struct.st_MDataArray*)

declare i32 @_ZL29MTensor_getMTensorInitializedPP13st_MDataArrayS0_Pxx(%struct.st_MDataArray**, %struct.st_MDataArray*, i64*, i64)

declare i32 @_ZL18WF_MTensor_getPartPP13st_MDataArrayS0_xPiPPv(%struct.st_MDataArray**, %struct.st_MDataArray*, i64, i32*, i8**)

declare i32 @_ZL18WF_MTensor_setPartPP13st_MDataArrayS0_xPiPPv(%struct.st_MDataArray**, %struct.st_MDataArray*, i64, i32*, i8**)

declare i64 (i8*, i8*, i64, i64*, i32)* @_ZL23WF_getUnaryMathFunctionii(i32, i32)

declare i64 (i8*, i8*, i8*, i64, i64*, i32)* @_ZL24WF_getBinaryMathFunctioniii(i32, i32, i32)

declare %struct.st_MDataArray* @_ZL19LoadRankZeroMTensorPvix(i8*, i32, i64)

declare i32 @_ZL26WF_MTensor_fillFromMTensorP13st_MDataArrayPS0_(%struct.st_MDataArray*, %struct.st_MDataArray**)

declare i32 @_ZL23interpret_fc_LP64_errorP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData*, i64, %union.MArgument*, i32*)

declare void @DeinitializeCompileLibraryFunctions()

declare void @WolframRTL_initialize(i64)

declare %struct.st_WolframLibraryData* @WolframLibraryData_new(i64)

declare void @WolframLibraryData_free(%struct.st_WolframLibraryData*)

declare i32 @MemoryLinkedList_visit(%struct.MemoryLinkedList_struct*, i32 (i8*, i8*)*, i8*)

declare %struct.MemoryLinkedList_struct* @MemoryLinkedList_newLink(i64)

declare %struct.MemoryLinkedList_struct* @MemoryLinkedList_copy(%struct.MemoryLinkedList_struct*, i8* (i8*)*)

declare void @MemoryLinkedList_delete(%struct.MemoryLinkedList_struct*, void (i8*)*)

declare %struct.MBag_struct* @MBag_new(i32)

declare void @MBag_delete(%struct.MBag_struct*)

declare void @MBag_stuff(%struct.MBag_struct*, i64, i8*)

declare i32 @MBag_toMTensor(%struct.st_MDataArray**, %struct.MBag_struct*)

declare i32 @MBag_part(i8*, %struct.MBag_struct*, %struct.part_ind*)

declare %struct.MBagArray_struct* @MBagArray_new()

declare void @MBagArray_cleanUp(%struct.MBagArray_struct*)

declare void @MBagArray_delete(%struct.MBagArray_struct*)

declare i64 @MBagArray_newBag(%struct.MBagArray_struct*, i32)

declare i32 @checkFloatingPointException(i8*, i32, i32)

declare %struct.RuntimeData_struct* @RuntimeData_new(i64)

declare void @RuntimeData_free(%struct.RuntimeData_struct*)

declare void @RuntimeData_cleanUp(%struct.RuntimeData_struct*, i32)

declare void @ClearDLLInitializedMTensors(%struct.RuntimeData_struct*, i32)

declare void @CompiledBlockRandomClear(%struct.RuntimeData_struct*)

declare void @FreeDLLAllocatedMemory(%struct.RuntimeData_struct*)

declare %struct.M_TENSOR_INITIALIZATION_DATA_STRUCT* @GetInitializedMTensors(%struct.st_WolframLibraryData*, i64)

declare void @ReleaseInitializedMTensors(%struct.M_TENSOR_INITIALIZATION_DATA_STRUCT*)

declare i8** @getDLLFunctionArgumentSpace(%struct.st_WolframLibraryData*, i64)

declare i32 @CompiledBagPart(i8*, i64, i32, i8*, %struct.RuntimeData_struct*)

declare i8* @CompiledGetRandomGeneratorName(i64)

declare i32 @CompiledSeedRandom(i64*, i64)

declare void @CompiledBlockRandom(%struct.RuntimeData_struct*, i32)

declare i32 (%struct.st_WolframLibraryData*, i64, %union.MArgument*, i32*)* @GetDLLFunctionPointer(i8*, i8*)

declare i32 @EvaluateExpression(%struct.st_WolframLibraryData*, i8*, i32, i64, i8*)

declare i32 @RegisterLibraryExpressionManager(i8*, void (%struct.st_WolframLibraryData*, i32, i64)*)

declare i32 @UnregisterLibraryExpressionManager(i8*)

declare i32 @ReleaseManagedLibraryExpression(i8*, i64)

declare i32 @RegisterLibraryCallbackManager(i8*, i32 (%struct.st_WolframLibraryData*, i64, %struct.st_MDataArray*)*)

declare i32 @UnregisterLibraryCallbackManager(i8*)

declare i32 @CallLibraryCallbackFunction(i64, i64, %union.MArgument*, i32*)

declare i32 @ReleaseLibraryCallbackFunction(i64)

declare i32 @evaluateFunctionExpression(%struct.st_WolframLibraryData*, i8*, i64, i64, i64, i32*, i8**, i32, i64, i8*)

declare i8* @getFunctionExpressionPointer(%struct.st_WolframLibraryData*, i8*)

declare %struct.MLINK_STRUCT* @getMathLink_DLL(%struct.st_WolframLibraryData*)

declare %struct.MLENV_STRUCT* @getMathLinkEnvironment_DLL(%struct.st_WolframLibraryData*)

declare i32 @processMathLink_DLL(%struct.MLINK_STRUCT*)

declare void @DLLMessageFunction(i8*)

declare i64 @countDLLSharedReferences(i8*)

declare i32 @removeDLLSharedReference(i8*)

declare void @removeAllDLLSharedReferences(i8*)

declare i32 @registerInputStreamMethod(i8*, void (%struct.st_MInputStream*, i8*, i8*)*, i32 (i8*, i8*)*, i8*, void (i8*)*)

declare i32 @unregisterInputStreamMethod(i8*)

declare i32 @registerOutputStreamMethod(i8*, void (%struct.st_MOutputStream*, i8*, i8*, i32)*, i32 (i8*, i8*)*, i8*, void (i8*)*)

declare i32 @unregisterOutputStreamMethod(i8*)

declare i64 @CreateAsynchronousTaskWithoutThread()

declare i64 @RemoveAsynchronousTask(i64)

declare i64 @CreateAsynchronousTaskWithThread(void (i64, i8*)*, i8*)

declare void @RaiseAsyncEvent(i64, i8*, %struct.st_DataStore*)

declare i32 @AsynchronousTaskAliveQ(i64)

declare i32 @AsynchronousTaskStartedQ(i64)

declare %struct.st_DataStore* @CreateDataStore()

declare void @DeleteDataStore(%struct.st_DataStore*)

declare %struct.st_DataStore* @CopyDataStore(%struct.st_DataStore*)

declare void @DataStore_addBoolean(%struct.st_DataStore*, i32)

declare void @DataStore_addInteger(%struct.st_DataStore*, i64)

declare void @DataStore_addReal(%struct.st_DataStore*, double)

declare void @DataStore_addComplex(%struct.st_DataStore*, double, double)

declare void @DataStore_addString(%struct.st_DataStore*, i8*)

declare void @DataStore_addMTensor(%struct.st_DataStore*, %struct.st_MDataArray*)

declare void @DataStore_addMNumericArray(%struct.st_DataStore*, %struct.st_MDataArray*)

declare void @DataStore_addMImage(%struct.st_DataStore*, %struct.IMAGEOBJ_ENTRY*)

declare void @DataStore_addMSparseArray(%struct.st_DataStore*, %struct.st_MSparseArray*)

declare void @DataStore_addDataStore(%struct.st_DataStore*, %struct.st_DataStore*)

declare void @DataStore_addNamedBoolean(%struct.st_DataStore*, i8*, i32)

declare void @DataStore_addNamedInteger(%struct.st_DataStore*, i8*, i64)

declare void @DataStore_addNamedReal(%struct.st_DataStore*, i8*, double)

declare void @DataStore_addNamedComplex(%struct.st_DataStore*, i8*, double, double)

declare void @DataStore_addNamedString(%struct.st_DataStore*, i8*, i8*)

declare void @DataStore_addNamedMTensor(%struct.st_DataStore*, i8*, %struct.st_MDataArray*)

declare void @DataStore_addNamedMNumericArray(%struct.st_DataStore*, i8*, %struct.st_MDataArray*)

declare void @DataStore_addNamedMImage(%struct.st_DataStore*, i8*, %struct.IMAGEOBJ_ENTRY*)

declare void @DataStore_addNamedMSparseArray(%struct.st_DataStore*, i8*, %struct.st_MSparseArray*)

declare void @DataStore_addNamedDataStore(%struct.st_DataStore*, i8*, %struct.st_DataStore*)

declare %struct.DataStoreNode_t* @DataStore_getFirstNode(%struct.st_DataStore*)

declare %struct.DataStoreNode_t* @DataStore_getLastNode(%struct.st_DataStore*)

declare i64 @DataStore_getLength(%struct.st_DataStore*)

declare %struct.DataStoreNode_t* @DataStoreNode_getNextNode(%struct.DataStoreNode_t*)

declare i32 @DataStoreNode_getDataType(%struct.DataStoreNode_t*)

declare i32 @DataStoreNode_getData(%struct.DataStoreNode_t*, %union.MArgument*)

declare i32 @DataStoreNode_getName(%struct.DataStoreNode_t*, i8**)

declare i32 @validatePathLib(i8*, i8)

declare i32 @protectedModeQ()

declare %struct.st_WolframImageLibrary_Functions* @getWolframImageLibraryFunctions(i64)

declare i32 @_ZL12MImage_new2Dxxx16MImage_Data_Type14MImage_CS_TypeiPP14IMAGEOBJ_ENTRY(i64, i64, i64, i32, i32, i32, %struct.IMAGEOBJ_ENTRY**)

declare i32 @_ZL12MImage_new3Dxxxx16MImage_Data_Type14MImage_CS_TypeiPP14IMAGEOBJ_ENTRY(i64, i64, i64, i64, i32, i32, i32, %struct.IMAGEOBJ_ENTRY**)

declare i32 @_ZL12MImage_cloneP14IMAGEOBJ_ENTRYPS0_(%struct.IMAGEOBJ_ENTRY*, %struct.IMAGEOBJ_ENTRY**)

declare void @_ZL11MImage_freeP14IMAGEOBJ_ENTRY(%struct.IMAGEOBJ_ENTRY*)

declare void @_ZL13MImage_disownP14IMAGEOBJ_ENTRY(%struct.IMAGEOBJ_ENTRY*)

declare void @_ZL16MImage_disownAllP14IMAGEOBJ_ENTRY(%struct.IMAGEOBJ_ENTRY*)

declare i64 @_ZL17MImage_shareCountP14IMAGEOBJ_ENTRY(%struct.IMAGEOBJ_ENTRY*)

declare i32 @_ZL18MImage_getDataTypeP14IMAGEOBJ_ENTRY(%struct.IMAGEOBJ_ENTRY*)

declare i64 @_ZL18MImage_getRowCountP14IMAGEOBJ_ENTRY(%struct.IMAGEOBJ_ENTRY*)

declare i64 @_ZL21MImage_getColumnCountP14IMAGEOBJ_ENTRY(%struct.IMAGEOBJ_ENTRY*)

declare i64 @_ZL20MImage_getSliceCountP14IMAGEOBJ_ENTRY(%struct.IMAGEOBJ_ENTRY*)

declare i64 @_ZL14MImage_getRankP14IMAGEOBJ_ENTRY(%struct.IMAGEOBJ_ENTRY*)

declare i64 @_ZL18MImage_getChannelsP14IMAGEOBJ_ENTRY(%struct.IMAGEOBJ_ENTRY*)

declare i32 @_ZL20MImage_alphaChannelQP14IMAGEOBJ_ENTRY(%struct.IMAGEOBJ_ENTRY*)

declare i32 @_ZL19MImage_interleavedQP14IMAGEOBJ_ENTRY(%struct.IMAGEOBJ_ENTRY*)

declare i32 @_ZL20MImage_getColorSpaceP14IMAGEOBJ_ENTRY(%struct.IMAGEOBJ_ENTRY*)

declare i64 @_ZL25MImage_getFlattenedLengthP14IMAGEOBJ_ENTRY(%struct.IMAGEOBJ_ENTRY*)

declare i32 @_ZL13MImage_getBitP14IMAGEOBJ_ENTRYPxxPa(%struct.IMAGEOBJ_ENTRY*, i64*, i64, i8*)

declare i32 @_ZL14MImage_getByteP14IMAGEOBJ_ENTRYPxxPh(%struct.IMAGEOBJ_ENTRY*, i64*, i64, i8*)

declare i32 @_ZL15MImage_getBit16P14IMAGEOBJ_ENTRYPxxPt(%struct.IMAGEOBJ_ENTRY*, i64*, i64, i16*)

declare i32 @_ZL16MImage_getReal32P14IMAGEOBJ_ENTRYPxxPf(%struct.IMAGEOBJ_ENTRY*, i64*, i64, float*)

declare i32 @_ZL14MImage_getRealP14IMAGEOBJ_ENTRYPxxPd(%struct.IMAGEOBJ_ENTRY*, i64*, i64, double*)

declare i32 @_ZL13MImage_setBitP14IMAGEOBJ_ENTRYPxxa(%struct.IMAGEOBJ_ENTRY*, i64*, i64, i8)

declare i32 @_ZL14MImage_setByteP14IMAGEOBJ_ENTRYPxxh(%struct.IMAGEOBJ_ENTRY*, i64*, i64, i8)

declare i32 @_ZL15MImage_setBit16P14IMAGEOBJ_ENTRYPxxt(%struct.IMAGEOBJ_ENTRY*, i64*, i64, i16)

declare i32 @_ZL16MImage_setReal32P14IMAGEOBJ_ENTRYPxxf(%struct.IMAGEOBJ_ENTRY*, i64*, i64, float)

declare i32 @_ZL14MImage_setRealP14IMAGEOBJ_ENTRYPxxd(%struct.IMAGEOBJ_ENTRY*, i64*, i64, double)

declare i8* @_ZL17MImage_getRawDataP14IMAGEOBJ_ENTRY(%struct.IMAGEOBJ_ENTRY*)

declare i8* @_ZL17MImage_getBitDataP14IMAGEOBJ_ENTRY(%struct.IMAGEOBJ_ENTRY*)

declare i8* @_ZL18MImage_getByteDataP14IMAGEOBJ_ENTRY(%struct.IMAGEOBJ_ENTRY*)

declare i16* @_ZL19MImage_getBit16DataP14IMAGEOBJ_ENTRY(%struct.IMAGEOBJ_ENTRY*)

declare float* @_ZL20MImage_getReal32DataP14IMAGEOBJ_ENTRY(%struct.IMAGEOBJ_ENTRY*)

declare double* @_ZL18MImage_getRealDataP14IMAGEOBJ_ENTRY(%struct.IMAGEOBJ_ENTRY*)

declare %struct.IMAGEOBJ_ENTRY* @_ZL18MImage_convertTypeP14IMAGEOBJ_ENTRY16MImage_Data_Typei(%struct.IMAGEOBJ_ENTRY*, i32, i32)

declare void @DeinitializeImageLibraryFunctions()

declare %struct.st_WolframNumericArrayLibrary_Functions* @getWolframNumericArrayLibraryFunctions(i64)

declare i32 @_ZL20MNumericArray_newDLL23MNumericArray_Data_TypexPKxPP13st_MDataArray(i32, i64, i64*, %struct.st_MDataArray**)

declare void @_ZL18MNumericArray_freeP13st_MDataArray(%struct.st_MDataArray*)

declare i32 @_ZL19MNumericArray_cloneP13st_MDataArrayPS0_(%struct.st_MDataArray*, %struct.st_MDataArray**)

declare void @_ZL20MNumericArray_disownP13st_MDataArray(%struct.st_MDataArray*)

declare void @_ZL23MNumericArray_disownAllP13st_MDataArray(%struct.st_MDataArray*)

declare i64 @_ZL24MNumericArray_shareCountP13st_MDataArray(%struct.st_MDataArray*)

declare i32 @_ZL24MNumericArray_getTypeDLLP13st_MDataArray(%struct.st_MDataArray*)

declare i64 @_ZL24MNumericArray_getRankDLLP13st_MDataArray(%struct.st_MDataArray*)

declare i64* @_ZL30MNumericArray_getDimensionsDLLP13st_MDataArray(%struct.st_MDataArray*)

declare i64 @_ZL32MNumericArray_getFlattenedLengthP13st_MDataArray(%struct.st_MDataArray*)

declare i8* @_ZL21MNumericArray_getDataP13st_MDataArray(%struct.st_MDataArray*)

declare i32 @_ZL25MNumericArray_convertTypePP13st_MDataArrayS0_23MNumericArray_Data_Type28MNumericArray_Convert_Methodd(%struct.st_MDataArray**, %struct.st_MDataArray*, i32, i32, double)

declare void @DeinitializeNumericArrayLibraryFunctions()

declare %struct.st_WolframRawArrayLibrary_Functions* @getWolframRawArrayLibraryFunctions(i64)

declare i32 @_ZL16MRawArray_newDLL19MRawArray_Data_TypexPKxPP13st_MDataArray(i32, i64, i64*, %struct.st_MDataArray**)

declare void @_ZL14MRawArray_freeP13st_MDataArray(%struct.st_MDataArray*)

declare i32 @_ZL15MRawArray_cloneP13st_MDataArrayPS0_(%struct.st_MDataArray*, %struct.st_MDataArray**)

declare void @_ZL16MRawArray_disownP13st_MDataArray(%struct.st_MDataArray*)

declare void @_ZL19MRawArray_disownAllP13st_MDataArray(%struct.st_MDataArray*)

declare i64 @_ZL20MRawArray_shareCountP13st_MDataArray(%struct.st_MDataArray*)

declare i32 @_ZL20MRawArray_getTypeDLLP13st_MDataArray(%struct.st_MDataArray*)

declare i64 @_ZL20MRawArray_getRankDLLP13st_MDataArray(%struct.st_MDataArray*)

declare i64* @_ZL26MRawArray_getDimensionsDLLP13st_MDataArray(%struct.st_MDataArray*)

declare i64 @_ZL28MRawArray_getFlattenedLengthP13st_MDataArray(%struct.st_MDataArray*)

declare i8* @_ZL17MRawArray_getDataP13st_MDataArray(%struct.st_MDataArray*)

declare %struct.st_MDataArray* @_ZL21MRawArray_convertTypeP13st_MDataArray19MRawArray_Data_Type(%struct.st_MDataArray*, i32)

declare void @DeinitializeRawArrayLibraryFunctions()

declare %struct.st_WolframSparseLibrary_Functions* @getWolframSparseLibraryFunctions(i64)

declare i32 @_ZL18MSparseArray_cloneP15st_MSparseArrayPS0_(%struct.st_MSparseArray*, %struct.st_MSparseArray**)

declare void @_ZL17MSparseArray_freeP15st_MSparseArray(%struct.st_MSparseArray*)

declare void @_ZL19MSparseArray_disownP15st_MSparseArray(%struct.st_MSparseArray*)

declare void @_ZL22MSparseArray_disownAllP15st_MSparseArray(%struct.st_MSparseArray*)

declare i64 @_ZL23MSparseArray_shareCountP15st_MSparseArray(%struct.st_MSparseArray*)

declare i64 @_ZL28MSparseArray_getRankFunctionP15st_MSparseArray(%struct.st_MSparseArray*)

declare i64* @_ZL29MTensor_GetDimensionsFunctionP15st_MSparseArray(%struct.st_MSparseArray*)

declare %struct.st_MDataArray** @_ZL37MSparseArray_getImplicitValueFunctionP15st_MSparseArray(%struct.st_MSparseArray*)

declare %struct.st_MDataArray** @_ZL38MSparseArray_getExplicitValuesFunctionP15st_MSparseArray(%struct.st_MSparseArray*)

declare %struct.st_MDataArray** @_ZL35MSparseArray_getRowPointersFunctionP15st_MSparseArray(%struct.st_MSparseArray*)

declare %struct.st_MDataArray** @_ZL37MSparseArray_getColumnIndicesFunctionP15st_MSparseArray(%struct.st_MSparseArray*)

declare i32 @_ZL41MSparseArray_getExplicitPositionsFunctionP15st_MSparseArrayPP13st_MDataArray(%struct.st_MSparseArray*, %struct.st_MDataArray**)

declare void @DeinitializeSparseLibraryFunctions()

declare %struct.st_WolframIOLibrary_Functions* @getWolframIOLibraryFunctions(i64)

declare void @DeinitializeIOLibraryFunctions()

declare void @_Z15InitRTLCompileri(i32)

declare %struct.st_MDataArray* @CreatePackedArray(i32, i64, i64*)

declare %struct.st_MDataArray* @Runtime_CreateNumericArray(i32, i64, i64*)

declare void @FreeMTensor(%struct.st_MDataArray*)

declare i64 @MTensor_RefCountIncrement(%struct.st_MDataArray*)

declare i64 @MTensor_RefCountDecrement(%struct.st_MDataArray*)

declare i32 @MTensor_getParts(%struct.st_MDataArray*, %struct.st_MDataArray**, %struct.st_MDataArray*, i64, i64, i32*, i8**)

declare i32 @MTensor_setParts(%struct.st_MDataArray*, %struct.st_MDataArray*, i64, i64, i32*, i8**)

declare i32 @MNumericArray_getParts(%struct.st_MDataArray*, %struct.st_MDataArray**, %struct.st_MDataArray*, i64, i64, i32*, i8**)

declare i32 @MTensorMathUnary(%struct.st_MDataArray**, %struct.st_MDataArray*, i32, i32)

declare i32 @MTensorMathBinary(%struct.st_MDataArray**, %struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i32 @VectorDot_I(i64*, i64*, i64, i64*, i64, i64, i32)

declare i32 @VectorDot_R64(double*, double*, i64, double*, i64, i64, i32)

declare i32 @VectorDot_CR64(i8*, i8*, i64, i8*, i64, i64, i32)

declare i32 @MTensorGeneralDot(%struct.st_MDataArray**, %struct.st_MDataArray*, %struct.st_MDataArray*, i32)

declare i64 @Runtime_IteratorCount_I_I_I_I(i64, i64, i64)

declare i64 @Runtime_IteratorCount_R_R_R_I(double, double, double)

declare i32 @Runtime_UniformRandomMIntegers(i64*, i64, i64, i64)

declare i32 @Runtime_UniformRandomMReals(double*, i64, double, double)

declare i32 @Runtime_UniformRandomMComplexes(i8*, i64, double, double, double, double)

declare i32 @Runtime_SeedRandom(i64, i64)

declare %struct.RandomGenerator_struct* @Runtime_GetCurrentRandomGenerator()

declare i64 (i64, %struct.RandomGenerator_struct*)* @Runtime_RandomGenerator_getBitFunction(%struct.RandomGenerator_struct*)

declare i32 @Runtime_MTensor_TakeDrop(%struct.st_MDataArray**, %struct.st_MDataArray*, i64, i8**, i32)

declare i32 @Runtime_MTensor_Partition(%struct.st_MDataArray**, %struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*)

declare i32 @Runtime_MTensor_Reverse(%struct.st_MDataArray**, %struct.st_MDataArray*)

declare i32 @Runtime_MTensor_Flatten(%struct.st_MDataArray**, %struct.st_MDataArray*, i64)

declare i32 @Runtime_MTensor_Rotate(%struct.st_MDataArray**, %struct.st_MDataArray*, i64, i64*)

declare i32 @Runtime_MTensor_12TransposeConjugate(%struct.st_MDataArray**, %struct.st_MDataArray*, i32)

declare i32 @Runtime_MTensor_ndimTransposeConjugate(%struct.st_MDataArray**, %struct.st_MDataArray*, %struct.st_MDataArray*, i32)

declare i32 @Runtime_MTensor_Sort(%struct.st_MDataArray**, %struct.st_MDataArray*)

declare i32 @Runtime_MTensor_computeSortPermutation(%struct.st_MDataArray*, %struct.st_MDataArray*)

declare i32 @_ZL22abortWatchCallbackInitPKc(i8*)

declare %struct.CompilerError_st* @CompilerError_null()

declare %struct.CompilerError_st* @New_CompilerError(i32, i32)

declare void @Delete_CompilerError(%struct.CompilerError_st*)

declare void @InitializeWolframRTL(i8*, i8*, i8*)

declare %struct.st_MDataArray* @getEternalMTensor()

declare void @_Z17init_compiler_rtli(i32)

declare void @Unwind_Resume_Wolfram(%struct._Unwind_Exception*)

declare void @_Unwind_Resume(%struct._Unwind_Exception*)

declare void @setExceptionStyle(i32)

declare i32 @getExceptionActive()

declare i32 @setExceptionActive(i32)

declare i32* @SetJumpStack_Push()

declare i32 @SetJumpStack_PushAux()

declare i32 @setjmp(i32*)

declare void @SetJumpStack_Pop()

declare %struct.CompilerError_st* @catchExceptionHandler(i32, i8**)

declare i32 @throwWolframException(i32)

declare void @longjmp(i32*, i32)

declare void @_ZN23WolframRuntimeExceptionD1Ev(%class.WolframRuntimeException*)

declare void @_ZNSt9exceptionD2Ev(%"class.std::exception"*)

declare void @_ZN23WolframRuntimeExceptionD0Ev(%class.WolframRuntimeException*)

declare i8* @_ZNK23WolframRuntimeException4whatEv(%class.WolframRuntimeException*)

declare void @_ZNSt3__19to_stringEi(%"class.std::__1::basic_string"*, i32)

declare void @_ZdlPv(i8*)

declare void @__cxa_call_unexpected(i8*)

declare i32* @getExceptionActiveHandle()

declare i32 @dummyFunction(i32)

declare i32* @getAbortWatchHandle()

declare i32 @checkAbortWatch()

declare i32 @checkAbortWatchThrow()

declare i32 @NonParallelDo_Closure(i8*, i8*, i64, i64)

declare i32 @NonParallelDo(i32 (i64)*, i64, i64)

declare %struct.st_MDataArray* @mtensor_null()

declare %struct.st_MDataArray* @mnumericarray_null()

declare i32 @testRealType(double, i32)

declare i8 @binary_abserr_Integer8_Integer8(i8, i8)

declare i8 @checked_binary_abserr_Integer8_Integer8(i8, i8)

declare i1 @_ZN3wrt9scalar_op6binary6detail16subtract_checkedIaEEbT_S4_PS4_(i8, i8, i8*)

declare i1 @_ZN3wrtL23check_integral_overflowIsaEENSt3__19enable_ifIXaasr18is_signed_integralIT_EE5valuesr11is_integralIT0_EE5valueEbE4typeERKS3_(i16)

declare i16 @binary_abserr_Integer16_Integer16(i16, i16)

declare i16 @checked_binary_abserr_Integer16_Integer16(i16, i16)

declare i1 @_ZN3wrt9scalar_op6binary6detail16subtract_checkedIsEEbT_S4_PS4_(i16, i16, i16*)

declare i1 @_ZN3wrtL23check_integral_overflowIisEENSt3__19enable_ifIXaasr18is_signed_integralIT_EE5valuesr11is_integralIT0_EE5valueEbE4typeERKS3_(i32)

declare i32 @binary_abserr_Integer32_Integer32(i32, i32)

declare i32 @checked_binary_abserr_Integer32_Integer32(i32, i32)

; Function Attrs: nounwind readnone speculatable
declare { i32, i1 } @llvm.ssub.with.overflow.i32(i32, i32) #0

declare i64 @binary_abserr_Integer64_Integer64(i64, i64)

declare i64 @checked_binary_abserr_Integer64_Integer64(i64, i64)

declare i8 @binary_abserr_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i8 @checked_binary_abserr_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i16 @binary_abserr_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i16 @checked_binary_abserr_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i32 @binary_abserr_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i32 @checked_binary_abserr_UnsignedInteger32_UnsignedInteger32(i32, i32)

; Function Attrs: nounwind readnone speculatable
declare { i32, i1 } @llvm.usub.with.overflow.i32(i32, i32) #0

declare i64 @binary_abserr_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i64 @checked_binary_abserr_UnsignedInteger64_UnsignedInteger64(i64, i64)

; Function Attrs: nounwind readnone speculatable
declare { i64, i1 } @llvm.usub.with.overflow.i64(i64, i64) #0

declare i16 @binary_abserr_Real16_Real16(i16, i16)

declare i16 @checked_binary_abserr_Real16_Real16(i16, i16)

declare float @binary_abserr_Real32_Real32(float, float)

; Function Attrs: nounwind readnone speculatable
declare float @llvm.fabs.f32(float) #0

declare float @checked_binary_abserr_Real32_Real32(float, float)

declare i32 @__fpclassifyf(float)

declare double @binary_abserr_Real64_Real64(double, double)

declare double @checked_binary_abserr_Real64_Real64(double, double)

declare float @binary_abserr_ComplexReal32_ComplexReal32(<2 x float>, <2 x float>)

declare float @hypotf(float, float)

declare float @checked_binary_abserr_ComplexReal32_ComplexReal32(<2 x float>, <2 x float>)

declare double @binary_abserr_ComplexReal64_ComplexReal64(double, double, double, double)

declare double @checked_binary_abserr_ComplexReal64_ComplexReal64(double, double, double, double)

declare i8 @binary_atan2_Integer8_Integer8(i8, i8)

declare i8 @checked_binary_atan2_Integer8_Integer8(i8, i8)

declare i16 @binary_atan2_Integer16_Integer16(i16, i16)

declare i16 @checked_binary_atan2_Integer16_Integer16(i16, i16)

declare i32 @binary_atan2_Integer32_Integer32(i32, i32)

declare i32 @checked_binary_atan2_Integer32_Integer32(i32, i32)

declare i64 @binary_atan2_Integer64_Integer64(i64, i64)

declare i64 @checked_binary_atan2_Integer64_Integer64(i64, i64)

declare i8 @binary_atan2_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i8 @checked_binary_atan2_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i16 @binary_atan2_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i16 @checked_binary_atan2_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i32 @binary_atan2_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i32 @checked_binary_atan2_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i64 @binary_atan2_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i64 @checked_binary_atan2_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i16 @binary_atan2_Real16_Real16(i16, i16)

declare float @atan2f(float, float)

declare i16 @checked_binary_atan2_Real16_Real16(i16, i16)

declare float @binary_atan2_Real32_Real32(float, float)

declare float @checked_binary_atan2_Real32_Real32(float, float)

declare double @binary_atan2_Real64_Real64(double, double)

declare double @checked_binary_atan2_Real64_Real64(double, double)

declare void @binary_atan2_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float)

declare <2 x float> @_ZN3wrt9scalar_op5unaryL8rsqrt_opINSt3__17complexIfEELNS_13runtime_flagsE0EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE64EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE(%"class.std::__1::complex.156"*)

; Function Attrs: nounwind readnone speculatable
declare float @llvm.log.f32(float) #0

; Function Attrs: nounwind readnone speculatable
declare float @llvm.sqrt.f32(float) #0

; Function Attrs: nounwind readnone speculatable
declare <2 x float> @llvm.fabs.v2f32(<2 x float>) #0

declare void @__assert_rtn(i8*, i8*, i32, i8*)

; Function Attrs: nounwind readnone speculatable
declare float @llvm.copysign.f32(float, float) #0

declare void @checked_binary_atan2_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float)

declare void @binary_atan2_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double)

declare { double, double } @_ZN3wrt9scalar_op5unaryL8rsqrt_opINSt3__17complexIdEELNS_13runtime_flagsE0EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE64EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE.176(%"class.std::__1::complex"*)

declare void @checked_binary_atan2_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double)

declare i8 @binary_bit_and_Integer8_Integer8(i8, i8)

declare i8 @checked_binary_bit_and_Integer8_Integer8(i8, i8)

declare i16 @binary_bit_and_Integer16_Integer16(i16, i16)

declare i16 @checked_binary_bit_and_Integer16_Integer16(i16, i16)

declare i32 @binary_bit_and_Integer32_Integer32(i32, i32)

declare i32 @checked_binary_bit_and_Integer32_Integer32(i32, i32)

declare i64 @binary_bit_and_Integer64_Integer64(i64, i64)

declare i64 @checked_binary_bit_and_Integer64_Integer64(i64, i64)

declare i8 @binary_bit_and_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i8 @checked_binary_bit_and_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i16 @binary_bit_and_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i16 @checked_binary_bit_and_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i32 @binary_bit_and_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i32 @checked_binary_bit_and_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i64 @binary_bit_and_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i64 @checked_binary_bit_and_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i16 @binary_bit_and_Real16_Real16(i16, i16)

declare i16 @checked_binary_bit_and_Real16_Real16(i16, i16)

declare float @binary_bit_and_Real32_Real32(float, float)

declare float @checked_binary_bit_and_Real32_Real32(float, float)

declare double @binary_bit_and_Real64_Real64(double, double)

declare double @checked_binary_bit_and_Real64_Real64(double, double)

declare <2 x float> @binary_bit_and_ComplexReal32_ComplexReal32(<2 x float>, <2 x float>)

declare <2 x float> @checked_binary_bit_and_ComplexReal32_ComplexReal32(<2 x float>, <2 x float>)

declare { double, double } @binary_bit_and_ComplexReal64_ComplexReal64(double, double, double, double)

declare { double, double } @checked_binary_bit_and_ComplexReal64_ComplexReal64(double, double, double, double)

declare i8 @binary_bit_or_Integer8_Integer8(i8, i8)

declare i8 @checked_binary_bit_or_Integer8_Integer8(i8, i8)

declare i16 @binary_bit_or_Integer16_Integer16(i16, i16)

declare i16 @checked_binary_bit_or_Integer16_Integer16(i16, i16)

declare i32 @binary_bit_or_Integer32_Integer32(i32, i32)

declare i32 @checked_binary_bit_or_Integer32_Integer32(i32, i32)

declare i64 @binary_bit_or_Integer64_Integer64(i64, i64)

declare i64 @checked_binary_bit_or_Integer64_Integer64(i64, i64)

declare i8 @binary_bit_or_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i8 @checked_binary_bit_or_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i16 @binary_bit_or_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i16 @checked_binary_bit_or_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i32 @binary_bit_or_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i32 @checked_binary_bit_or_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i64 @binary_bit_or_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i64 @checked_binary_bit_or_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i16 @binary_bit_or_Real16_Real16(i16, i16)

declare i16 @checked_binary_bit_or_Real16_Real16(i16, i16)

declare float @binary_bit_or_Real32_Real32(float, float)

declare float @checked_binary_bit_or_Real32_Real32(float, float)

declare double @binary_bit_or_Real64_Real64(double, double)

declare double @checked_binary_bit_or_Real64_Real64(double, double)

declare <2 x float> @binary_bit_or_ComplexReal32_ComplexReal32(<2 x float>, <2 x float>)

declare <2 x float> @checked_binary_bit_or_ComplexReal32_ComplexReal32(<2 x float>, <2 x float>)

declare { double, double } @binary_bit_or_ComplexReal64_ComplexReal64(double, double, double, double)

declare { double, double } @checked_binary_bit_or_ComplexReal64_ComplexReal64(double, double, double, double)

declare i8 @binary_bit_shiftleft_Integer8_Integer8(i8, i8)

declare i8 @checked_binary_bit_shiftleft_Integer8_Integer8(i8, i8)

declare i16 @binary_bit_shiftleft_Integer16_Integer16(i16, i16)

declare i16 @checked_binary_bit_shiftleft_Integer16_Integer16(i16, i16)

declare i32 @binary_bit_shiftleft_Integer32_Integer32(i32, i32)

declare i32 @checked_binary_bit_shiftleft_Integer32_Integer32(i32, i32)

declare i64 @binary_bit_shiftleft_Integer64_Integer64(i64, i64)

declare i64 @checked_binary_bit_shiftleft_Integer64_Integer64(i64, i64)

declare i8 @binary_bit_shiftleft_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i8 @checked_binary_bit_shiftleft_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i16 @binary_bit_shiftleft_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i16 @checked_binary_bit_shiftleft_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i32 @binary_bit_shiftleft_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i32 @checked_binary_bit_shiftleft_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i64 @binary_bit_shiftleft_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i64 @checked_binary_bit_shiftleft_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i16 @binary_bit_shiftleft_Real16_Real16(i16, i16)

declare i16 @checked_binary_bit_shiftleft_Real16_Real16(i16, i16)

declare float @binary_bit_shiftleft_Real32_Real32(float, float)

declare float @checked_binary_bit_shiftleft_Real32_Real32(float, float)

declare double @binary_bit_shiftleft_Real64_Real64(double, double)

declare double @checked_binary_bit_shiftleft_Real64_Real64(double, double)

declare void @binary_bit_shiftleft_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float)

declare void @checked_binary_bit_shiftleft_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float)

declare void @binary_bit_shiftleft_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double)

declare void @checked_binary_bit_shiftleft_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double)

declare i8 @binary_bit_shiftright_Integer8_Integer8(i8, i8)

declare i8 @checked_binary_bit_shiftright_Integer8_Integer8(i8, i8)

declare i16 @binary_bit_shiftright_Integer16_Integer16(i16, i16)

declare i16 @checked_binary_bit_shiftright_Integer16_Integer16(i16, i16)

declare i32 @binary_bit_shiftright_Integer32_Integer32(i32, i32)

declare i32 @checked_binary_bit_shiftright_Integer32_Integer32(i32, i32)

declare i64 @binary_bit_shiftright_Integer64_Integer64(i64, i64)

declare i64 @checked_binary_bit_shiftright_Integer64_Integer64(i64, i64)

declare i8 @binary_bit_shiftright_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i8 @checked_binary_bit_shiftright_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i16 @binary_bit_shiftright_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i16 @checked_binary_bit_shiftright_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i32 @binary_bit_shiftright_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i32 @checked_binary_bit_shiftright_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i64 @binary_bit_shiftright_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i64 @checked_binary_bit_shiftright_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i16 @binary_bit_shiftright_Real16_Real16(i16, i16)

declare i16 @checked_binary_bit_shiftright_Real16_Real16(i16, i16)

declare float @binary_bit_shiftright_Real32_Real32(float, float)

declare float @checked_binary_bit_shiftright_Real32_Real32(float, float)

declare double @binary_bit_shiftright_Real64_Real64(double, double)

declare double @checked_binary_bit_shiftright_Real64_Real64(double, double)

declare void @binary_bit_shiftright_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float)

declare void @checked_binary_bit_shiftright_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float)

declare void @binary_bit_shiftright_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double)

declare void @checked_binary_bit_shiftright_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double)

declare i8 @binary_bit_xor_Integer8_Integer8(i8, i8)

declare i8 @checked_binary_bit_xor_Integer8_Integer8(i8, i8)

declare i16 @binary_bit_xor_Integer16_Integer16(i16, i16)

declare i16 @checked_binary_bit_xor_Integer16_Integer16(i16, i16)

declare i32 @binary_bit_xor_Integer32_Integer32(i32, i32)

declare i32 @checked_binary_bit_xor_Integer32_Integer32(i32, i32)

declare i64 @binary_bit_xor_Integer64_Integer64(i64, i64)

declare i64 @checked_binary_bit_xor_Integer64_Integer64(i64, i64)

declare i8 @binary_bit_xor_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i8 @checked_binary_bit_xor_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i16 @binary_bit_xor_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i16 @checked_binary_bit_xor_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i32 @binary_bit_xor_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i32 @checked_binary_bit_xor_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i64 @binary_bit_xor_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i64 @checked_binary_bit_xor_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i16 @binary_bit_xor_Real16_Real16(i16, i16)

declare i16 @checked_binary_bit_xor_Real16_Real16(i16, i16)

declare float @binary_bit_xor_Real32_Real32(float, float)

declare float @checked_binary_bit_xor_Real32_Real32(float, float)

declare double @binary_bit_xor_Real64_Real64(double, double)

declare double @checked_binary_bit_xor_Real64_Real64(double, double)

declare <2 x float> @binary_bit_xor_ComplexReal32_ComplexReal32(<2 x float>, <2 x float>)

declare <2 x float> @checked_binary_bit_xor_ComplexReal32_ComplexReal32(<2 x float>, <2 x float>)

declare { double, double } @binary_bit_xor_ComplexReal64_ComplexReal64(double, double, double, double)

declare { double, double } @checked_binary_bit_xor_ComplexReal64_ComplexReal64(double, double, double, double)

declare i8 @binary_chop_Integer8_Integer8(i8, i8)

declare i8 @checked_binary_chop_Integer8_Integer8(i8, i8)

declare i16 @binary_chop_Integer16_Integer16(i16, i16)

declare i16 @checked_binary_chop_Integer16_Integer16(i16, i16)

declare i32 @binary_chop_Integer32_Integer32(i32, i32)

declare i32 @checked_binary_chop_Integer32_Integer32(i32, i32)

declare i64 @binary_chop_Integer64_Integer64(i64, i64)

declare i64 @checked_binary_chop_Integer64_Integer64(i64, i64)

declare i8 @binary_chop_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i8 @checked_binary_chop_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i16 @binary_chop_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i16 @checked_binary_chop_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i32 @binary_chop_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i32 @checked_binary_chop_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i64 @binary_chop_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i64 @checked_binary_chop_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i16 @binary_chop_Real16_Real16(i16, i16)

declare i16 @checked_binary_chop_Real16_Real16(i16, i16)

declare float @binary_chop_Real32_Real32(float, float)

declare float @checked_binary_chop_Real32_Real32(float, float)

declare double @binary_chop_Real64_Real64(double, double)

declare double @checked_binary_chop_Real64_Real64(double, double)

declare void @binary_chop_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float)

declare void @checked_binary_chop_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float)

declare void @binary_chop_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double)

declare void @checked_binary_chop_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double)

declare i8 @binary_divide_Integer8_Integer8(i8, i8)

declare i8 @checked_binary_divide_Integer8_Integer8(i8, i8)

declare i16 @binary_divide_Integer16_Integer16(i16, i16)

declare i16 @checked_binary_divide_Integer16_Integer16(i16, i16)

declare i32 @binary_divide_Integer32_Integer32(i32, i32)

declare i32 @checked_binary_divide_Integer32_Integer32(i32, i32)

declare i64 @binary_divide_Integer64_Integer64(i64, i64)

declare i64 @checked_binary_divide_Integer64_Integer64(i64, i64)

declare i8 @binary_divide_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i8 @checked_binary_divide_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i16 @binary_divide_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i16 @checked_binary_divide_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i32 @binary_divide_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i32 @checked_binary_divide_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i64 @binary_divide_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i64 @checked_binary_divide_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i16 @binary_divide_Real16_Real16(i16, i16)

declare i16 @checked_binary_divide_Real16_Real16(i16, i16)

declare float @binary_divide_Real32_Real32(float, float)

declare float @checked_binary_divide_Real32_Real32(float, float)

declare double @binary_divide_Real64_Real64(double, double)

declare double @checked_binary_divide_Real64_Real64(double, double)

declare void @binary_divide_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float)

declare <2 x float> @_ZNSt3__1dvIfEENS_7complexIT_EERKS3_S5_(%"class.std::__1::complex.156"*, %"class.std::__1::complex.156"*)

; Function Attrs: nounwind readnone speculatable
declare float @llvm.maxnum.f32(float, float) #0

declare float @logbf(float)

declare float @scalbnf(float, i32)

; Function Attrs: nounwind readnone speculatable
declare <2 x float> @llvm.copysign.v2f32(<2 x float>, <2 x float>) #0

declare void @checked_binary_divide_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float)

declare void @binary_divide_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double)

declare void @checked_binary_divide_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double)

declare i64 @binary_fpexception_Integer8_Integer8(i8, i8)

declare i64 @checked_binary_fpexception_Integer8_Integer8(i8, i8)

declare i64 @binary_fpexception_Integer16_Integer16(i16, i16)

declare i64 @checked_binary_fpexception_Integer16_Integer16(i16, i16)

declare i64 @binary_fpexception_Integer32_Integer32(i32, i32)

declare i64 @checked_binary_fpexception_Integer32_Integer32(i32, i32)

declare i64 @binary_fpexception_Integer64_Integer64(i64, i64)

declare i64 @checked_binary_fpexception_Integer64_Integer64(i64, i64)

declare i64 @binary_fpexception_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i64 @checked_binary_fpexception_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i64 @binary_fpexception_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i64 @checked_binary_fpexception_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i64 @binary_fpexception_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i64 @checked_binary_fpexception_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i64 @binary_fpexception_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i64 @checked_binary_fpexception_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i64 @binary_fpexception_Real16_Real16(i16, i16)

declare i64 @checked_binary_fpexception_Real16_Real16(i16, i16)

declare i64 @binary_fpexception_Real32_Real32(float, float)

declare i64 @checked_binary_fpexception_Real32_Real32(float, float)

declare i64 @binary_fpexception_Real64_Real64(double, double)

declare i64 @checked_binary_fpexception_Real64_Real64(double, double)

declare i64 @binary_fpexception_ComplexReal32_ComplexReal32(<2 x float>, <2 x float>)

declare i64 @checked_binary_fpexception_ComplexReal32_ComplexReal32(<2 x float>, <2 x float>)

declare i64 @binary_fpexception_ComplexReal64_ComplexReal64(double, double, double, double)

declare i64 @checked_binary_fpexception_ComplexReal64_ComplexReal64(double, double, double, double)

declare i8 @binary_intexp_Integer8_Integer8(i8, i8)

declare i64 @div(i32, i32)

declare i8 @checked_binary_intexp_Integer8_Integer8(i8, i8)

declare i16 @binary_intexp_Integer16_Integer16(i16, i16)

declare i16 @checked_binary_intexp_Integer16_Integer16(i16, i16)

declare i32 @binary_intexp_Integer32_Integer32(i32, i32)

; Function Attrs: nounwind readnone speculatable
declare i32 @llvm.cttz.i32(i32, i1) #0

declare i32 @checked_binary_intexp_Integer32_Integer32(i32, i32)

declare i64 @binary_intexp_Integer64_Integer64(i64, i64)

declare i64 @checked_binary_intexp_Integer64_Integer64(i64, i64)

declare i8 @binary_intexp_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i8 @checked_binary_intexp_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i16 @binary_intexp_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i16 @checked_binary_intexp_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i32 @binary_intexp_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i32 @checked_binary_intexp_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i64 @binary_intexp_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i64 @checked_binary_intexp_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i16 @binary_intexp_Real16_Real16(i16, i16)

declare i16 @checked_binary_intexp_Real16_Real16(i16, i16)

declare float @binary_intexp_Real32_Real32(float, float)

declare float @checked_binary_intexp_Real32_Real32(float, float)

declare double @binary_intexp_Real64_Real64(double, double)

declare double @checked_binary_intexp_Real64_Real64(double, double)

declare <2 x float> @binary_intexp_ComplexReal32_ComplexReal32(<2 x float>, <2 x float>)

declare <2 x float> @checked_binary_intexp_ComplexReal32_ComplexReal32(<2 x float>, <2 x float>)

declare { double, double } @binary_intexp_ComplexReal64_ComplexReal64(double, double, double, double)

declare { double, double } @checked_binary_intexp_ComplexReal64_ComplexReal64(double, double, double, double)

declare i8 @binary_intlen_Integer8_Integer8(i8, i8)

declare i8 @checked_binary_intlen_Integer8_Integer8(i8, i8)

declare i16 @binary_intlen_Integer16_Integer16(i16, i16)

declare i16 @checked_binary_intlen_Integer16_Integer16(i16, i16)

declare i32 @binary_intlen_Integer32_Integer32(i32, i32)

; Function Attrs: nounwind readnone speculatable
declare i32 @llvm.ctlz.i32(i32, i1) #0

declare i32 @checked_binary_intlen_Integer32_Integer32(i32, i32)

declare i64 @binary_intlen_Integer64_Integer64(i64, i64)

declare i64 @checked_binary_intlen_Integer64_Integer64(i64, i64)

declare i8 @binary_intlen_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i8 @checked_binary_intlen_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i16 @binary_intlen_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i16 @checked_binary_intlen_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i32 @binary_intlen_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i32 @checked_binary_intlen_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i64 @binary_intlen_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i64 @checked_binary_intlen_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i16 @binary_intlen_Real16_Real16(i16, i16)

declare i16 @checked_binary_intlen_Real16_Real16(i16, i16)

declare float @binary_intlen_Real32_Real32(float, float)

declare float @checked_binary_intlen_Real32_Real32(float, float)

declare double @binary_intlen_Real64_Real64(double, double)

declare double @checked_binary_intlen_Real64_Real64(double, double)

declare <2 x float> @binary_intlen_ComplexReal32_ComplexReal32(<2 x float>, <2 x float>)

declare <2 x float> @checked_binary_intlen_ComplexReal32_ComplexReal32(<2 x float>, <2 x float>)

declare { double, double } @binary_intlen_ComplexReal64_ComplexReal64(double, double, double, double)

declare { double, double } @checked_binary_intlen_ComplexReal64_ComplexReal64(double, double, double, double)

declare i8 @binary_log_Integer8_Integer8(i8, i8)

declare i8 @checked_binary_log_Integer8_Integer8(i8, i8)

declare i16 @binary_log_Integer16_Integer16(i16, i16)

declare i16 @checked_binary_log_Integer16_Integer16(i16, i16)

declare i32 @binary_log_Integer32_Integer32(i32, i32)

declare i32 @checked_binary_log_Integer32_Integer32(i32, i32)

declare i64 @binary_log_Integer64_Integer64(i64, i64)

declare i64 @checked_binary_log_Integer64_Integer64(i64, i64)

declare i8 @binary_log_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i8 @checked_binary_log_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i16 @binary_log_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i16 @checked_binary_log_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i32 @binary_log_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i32 @checked_binary_log_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i64 @binary_log_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i64 @checked_binary_log_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i16 @binary_log_Real16_Real16(i16, i16)

declare i16 @checked_binary_log_Real16_Real16(i16, i16)

declare float @binary_log_Real32_Real32(float, float)

declare float @checked_binary_log_Real32_Real32(float, float)

declare double @binary_log_Real64_Real64(double, double)

declare double @checked_binary_log_Real64_Real64(double, double)

declare void @binary_log_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float)

declare void @checked_binary_log_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float)

declare <2 x float> @_ZN3wrt9scalar_op6binaryL9divide_opINSt3__17complexIfEES5_LNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE9EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE(%"class.std::__1::complex.156"*, %"class.std::__1::complex.156"*)

declare void @binary_log_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double)

declare void @checked_binary_log_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double)

declare { double, double } @_ZN3wrt9scalar_op6binaryL9divide_opINSt3__17complexIdEES5_LNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE9EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE.177(%"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare i8 @binary_maxabs_Integer8_Integer8(i8, i8)

declare i8 @checked_binary_maxabs_Integer8_Integer8(i8, i8)

declare i16 @binary_maxabs_Integer16_Integer16(i16, i16)

declare i16 @checked_binary_maxabs_Integer16_Integer16(i16, i16)

declare i32 @binary_maxabs_Integer32_Integer32(i32, i32)

declare i32 @checked_binary_maxabs_Integer32_Integer32(i32, i32)

declare i64 @binary_maxabs_Integer64_Integer64(i64, i64)

declare i64 @checked_binary_maxabs_Integer64_Integer64(i64, i64)

declare i8 @binary_maxabs_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i8 @checked_binary_maxabs_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i16 @binary_maxabs_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i16 @checked_binary_maxabs_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i32 @binary_maxabs_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i32 @checked_binary_maxabs_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i64 @binary_maxabs_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i64 @checked_binary_maxabs_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i16 @binary_maxabs_Real16_Real16(i16, i16)

declare i16 @checked_binary_maxabs_Real16_Real16(i16, i16)

declare float @binary_maxabs_Real32_Real32(float, float)

declare float @checked_binary_maxabs_Real32_Real32(float, float)

declare double @binary_maxabs_Real64_Real64(double, double)

declare double @checked_binary_maxabs_Real64_Real64(double, double)

declare float @binary_maxabs_ComplexReal32_ComplexReal32(<2 x float>, <2 x float>)

declare float @checked_binary_maxabs_ComplexReal32_ComplexReal32(<2 x float>, <2 x float>)

declare double @binary_maxabs_ComplexReal64_ComplexReal64(double, double, double, double)

declare double @checked_binary_maxabs_ComplexReal64_ComplexReal64(double, double, double, double)

declare i8 @binary_max_Integer8_Integer8(i8, i8)

declare i8 @checked_binary_max_Integer8_Integer8(i8, i8)

declare i16 @binary_max_Integer16_Integer16(i16, i16)

declare i16 @checked_binary_max_Integer16_Integer16(i16, i16)

declare i32 @binary_max_Integer32_Integer32(i32, i32)

declare i32 @checked_binary_max_Integer32_Integer32(i32, i32)

declare i64 @binary_max_Integer64_Integer64(i64, i64)

declare i64 @checked_binary_max_Integer64_Integer64(i64, i64)

declare i8 @binary_max_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i8 @checked_binary_max_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i16 @binary_max_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i16 @checked_binary_max_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i32 @binary_max_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i32 @checked_binary_max_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i64 @binary_max_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i64 @checked_binary_max_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i16 @binary_max_Real16_Real16(i16, i16)

declare i16 @checked_binary_max_Real16_Real16(i16, i16)

declare float @binary_max_Real32_Real32(float, float)

declare float @checked_binary_max_Real32_Real32(float, float)

declare double @binary_max_Real64_Real64(double, double)

declare double @checked_binary_max_Real64_Real64(double, double)

declare <2 x float> @binary_max_ComplexReal32_ComplexReal32(<2 x float>, <2 x float>)

declare <2 x float> @checked_binary_max_ComplexReal32_ComplexReal32(<2 x float>, <2 x float>)

declare { double, double } @binary_max_ComplexReal64_ComplexReal64(double, double, double, double)

declare { double, double } @checked_binary_max_ComplexReal64_ComplexReal64(double, double, double, double)

declare i8 @binary_min_Integer8_Integer8(i8, i8)

declare i8 @checked_binary_min_Integer8_Integer8(i8, i8)

declare i16 @binary_min_Integer16_Integer16(i16, i16)

declare i16 @checked_binary_min_Integer16_Integer16(i16, i16)

declare i32 @binary_min_Integer32_Integer32(i32, i32)

declare i32 @checked_binary_min_Integer32_Integer32(i32, i32)

declare i64 @binary_min_Integer64_Integer64(i64, i64)

declare i64 @checked_binary_min_Integer64_Integer64(i64, i64)

declare i8 @binary_min_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i8 @checked_binary_min_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i16 @binary_min_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i16 @checked_binary_min_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i32 @binary_min_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i32 @checked_binary_min_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i64 @binary_min_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i64 @checked_binary_min_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i16 @binary_min_Real16_Real16(i16, i16)

declare i16 @checked_binary_min_Real16_Real16(i16, i16)

declare float @binary_min_Real32_Real32(float, float)

; Function Attrs: nounwind readnone speculatable
declare float @llvm.minnum.f32(float, float) #0

declare float @checked_binary_min_Real32_Real32(float, float)

declare double @binary_min_Real64_Real64(double, double)

declare double @checked_binary_min_Real64_Real64(double, double)

declare <2 x float> @binary_min_ComplexReal32_ComplexReal32(<2 x float>, <2 x float>)

declare <2 x float> @checked_binary_min_ComplexReal32_ComplexReal32(<2 x float>, <2 x float>)

declare { double, double } @binary_min_ComplexReal64_ComplexReal64(double, double, double, double)

declare { double, double } @checked_binary_min_ComplexReal64_ComplexReal64(double, double, double, double)

declare i8 @binary_mod_Integer8_Integer8(i8, i8)

declare i8 @checked_binary_mod_Integer8_Integer8(i8, i8)

declare i16 @binary_mod_Integer16_Integer16(i16, i16)

declare i16 @checked_binary_mod_Integer16_Integer16(i16, i16)

declare i32 @binary_mod_Integer32_Integer32(i32, i32)

declare i32 @checked_binary_mod_Integer32_Integer32(i32, i32)

declare i64 @binary_mod_Integer64_Integer64(i64, i64)

declare i64 @checked_binary_mod_Integer64_Integer64(i64, i64)

declare i8 @binary_mod_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i8 @checked_binary_mod_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i16 @binary_mod_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i16 @checked_binary_mod_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i32 @binary_mod_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i32 @checked_binary_mod_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i64 @binary_mod_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i64 @checked_binary_mod_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i16 @binary_mod_Real16_Real16(i16, i16)

declare i16 @checked_binary_mod_Real16_Real16(i16, i16)

declare float @binary_mod_Real32_Real32(float, float)

declare float @checked_binary_mod_Real32_Real32(float, float)

declare double @binary_mod_Real64_Real64(double, double)

declare double @checked_binary_mod_Real64_Real64(double, double)

declare void @binary_mod_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float)

declare i32 @fegetround()

; Function Attrs: nounwind readnone speculatable
declare float @llvm.nearbyint.f32(float) #0

declare void @checked_binary_mod_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float)

declare void @binary_mod_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double)

declare void @checked_binary_mod_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double)

declare i8 @binary_plus_Integer8_Integer8(i8, i8)

declare i8 @checked_binary_plus_Integer8_Integer8(i8, i8)

; Function Attrs: nounwind readnone speculatable
declare { i8, i1 } @llvm.sadd.with.overflow.i8(i8, i8) #0

declare i16 @binary_plus_Integer16_Integer16(i16, i16)

declare i16 @checked_binary_plus_Integer16_Integer16(i16, i16)

; Function Attrs: nounwind readnone speculatable
declare { i16, i1 } @llvm.sadd.with.overflow.i16(i16, i16) #0

declare i32 @binary_plus_Integer32_Integer32(i32, i32)

declare i32 @checked_binary_plus_Integer32_Integer32(i32, i32)

; Function Attrs: nounwind readnone speculatable
declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32) #0

declare i64 @binary_plus_Integer64_Integer64(i64, i64)

declare i64 @checked_binary_plus_Integer64_Integer64(i64, i64)

declare i8 @binary_plus_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i8 @checked_binary_plus_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i16 @binary_plus_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i16 @checked_binary_plus_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i32 @binary_plus_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i32 @checked_binary_plus_UnsignedInteger32_UnsignedInteger32(i32, i32)

; Function Attrs: nounwind readnone speculatable
declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32) #0

declare i64 @binary_plus_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i64 @checked_binary_plus_UnsignedInteger64_UnsignedInteger64(i64, i64)

; Function Attrs: nounwind readnone speculatable
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64) #0

declare i16 @binary_plus_Real16_Real16(i16, i16)

declare i16 @checked_binary_plus_Real16_Real16(i16, i16)

declare float @binary_plus_Real32_Real32(float, float)

declare float @checked_binary_plus_Real32_Real32(float, float)

declare double @binary_plus_Real64_Real64(double, double)

declare double @checked_binary_plus_Real64_Real64(double, double)

declare void @binary_plus_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float)

declare void @checked_binary_plus_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float)

declare void @binary_plus_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double)

declare void @checked_binary_plus_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double)

declare i8 @binary_quotient_Integer8_Integer8(i8, i8)

declare i8 @checked_binary_quotient_Integer8_Integer8(i8, i8)

declare i16 @binary_quotient_Integer16_Integer16(i16, i16)

declare i16 @checked_binary_quotient_Integer16_Integer16(i16, i16)

declare i32 @binary_quotient_Integer32_Integer32(i32, i32)

declare i32 @checked_binary_quotient_Integer32_Integer32(i32, i32)

declare i64 @binary_quotient_Integer64_Integer64(i64, i64)

declare i64 @checked_binary_quotient_Integer64_Integer64(i64, i64)

declare i8 @binary_quotient_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i8 @checked_binary_quotient_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i16 @binary_quotient_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i16 @checked_binary_quotient_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i32 @binary_quotient_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i32 @checked_binary_quotient_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i64 @binary_quotient_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i64 @checked_binary_quotient_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i64 @binary_quotient_Real16_Real16(i16, i16)

declare i64 @checked_binary_quotient_Real16_Real16(i16, i16)

declare i64 @binary_quotient_Real32_Real32(float, float)

; Function Attrs: nounwind readnone speculatable
declare float @llvm.floor.f32(float) #0

declare i64 @checked_binary_quotient_Real32_Real32(float, float)

declare i64 @binary_quotient_Real64_Real64(double, double)

declare i64 @checked_binary_quotient_Real64_Real64(double, double)

declare i64 @binary_quotient_ComplexReal32_ComplexReal32(<2 x float>, <2 x float>)

declare i64 @checked_binary_quotient_ComplexReal32_ComplexReal32(<2 x float>, <2 x float>)

declare i64 @binary_quotient_ComplexReal64_ComplexReal64(double, double, double, double)

declare i64 @checked_binary_quotient_ComplexReal64_ComplexReal64(double, double, double, double)

declare i8 @binary_root_Integer8_Integer8(i8, i8)

declare i8 @checked_binary_root_Integer8_Integer8(i8, i8)

declare i16 @binary_root_Integer16_Integer16(i16, i16)

declare i16 @checked_binary_root_Integer16_Integer16(i16, i16)

declare i32 @binary_root_Integer32_Integer32(i32, i32)

declare i32 @checked_binary_root_Integer32_Integer32(i32, i32)

declare i64 @binary_root_Integer64_Integer64(i64, i64)

declare i64 @checked_binary_root_Integer64_Integer64(i64, i64)

declare i8 @binary_root_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i8 @checked_binary_root_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i16 @binary_root_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i16 @checked_binary_root_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i32 @binary_root_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i32 @checked_binary_root_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i64 @binary_root_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i64 @checked_binary_root_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i16 @binary_root_Real16_Real16(i16, i16)

declare i16 @checked_binary_root_Real16_Real16(i16, i16)

declare float @binary_root_Real32_Real32(float, float)

declare float @checked_binary_root_Real32_Real32(float, float)

declare double @binary_root_Real64_Real64(double, double)

declare double @checked_binary_root_Real64_Real64(double, double)

declare <2 x float> @binary_root_ComplexReal32_ComplexReal32(<2 x float>, <2 x float>)

declare <2 x float> @checked_binary_root_ComplexReal32_ComplexReal32(<2 x float>, <2 x float>)

declare { double, double } @binary_root_ComplexReal64_ComplexReal64(double, double, double, double)

declare { double, double } @checked_binary_root_ComplexReal64_ComplexReal64(double, double, double, double)

declare i8 @binary_relerr_Integer8_Integer8(i8, i8)

declare i8 @checked_binary_relerr_Integer8_Integer8(i8, i8)

declare i16 @binary_relerr_Integer16_Integer16(i16, i16)

declare i16 @checked_binary_relerr_Integer16_Integer16(i16, i16)

declare i32 @binary_relerr_Integer32_Integer32(i32, i32)

declare i32 @checked_binary_relerr_Integer32_Integer32(i32, i32)

declare i64 @binary_relerr_Integer64_Integer64(i64, i64)

declare i64 @checked_binary_relerr_Integer64_Integer64(i64, i64)

declare i8 @binary_relerr_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i8 @checked_binary_relerr_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i16 @binary_relerr_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i16 @checked_binary_relerr_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i32 @binary_relerr_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i32 @checked_binary_relerr_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i64 @binary_relerr_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i64 @checked_binary_relerr_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i16 @binary_relerr_Real16_Real16(i16, i16)

declare i16 @checked_binary_relerr_Real16_Real16(i16, i16)

declare float @binary_relerr_Real32_Real32(float, float)

declare float @checked_binary_relerr_Real32_Real32(float, float)

declare double @binary_relerr_Real64_Real64(double, double)

declare double @checked_binary_relerr_Real64_Real64(double, double)

declare float @binary_relerr_ComplexReal32_ComplexReal32(<2 x float>, <2 x float>)

declare float @checked_binary_relerr_ComplexReal32_ComplexReal32(<2 x float>, <2 x float>)

declare double @binary_relerr_ComplexReal64_ComplexReal64(double, double, double, double)

declare double @checked_binary_relerr_ComplexReal64_ComplexReal64(double, double, double, double)

declare i8 @binary_subtract_Integer8_Integer8(i8, i8)

declare i8 @checked_binary_subtract_Integer8_Integer8(i8, i8)

declare i16 @binary_subtract_Integer16_Integer16(i16, i16)

declare i16 @checked_binary_subtract_Integer16_Integer16(i16, i16)

declare i32 @binary_subtract_Integer32_Integer32(i32, i32)

declare i32 @checked_binary_subtract_Integer32_Integer32(i32, i32)

declare i64 @binary_subtract_Integer64_Integer64(i64, i64)

declare i64 @checked_binary_subtract_Integer64_Integer64(i64, i64)

declare i8 @binary_subtract_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i8 @checked_binary_subtract_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i16 @binary_subtract_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i16 @checked_binary_subtract_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i32 @binary_subtract_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i32 @checked_binary_subtract_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i64 @binary_subtract_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i64 @checked_binary_subtract_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i16 @binary_subtract_Real16_Real16(i16, i16)

declare i16 @checked_binary_subtract_Real16_Real16(i16, i16)

declare float @binary_subtract_Real32_Real32(float, float)

declare float @checked_binary_subtract_Real32_Real32(float, float)

declare double @binary_subtract_Real64_Real64(double, double)

declare double @checked_binary_subtract_Real64_Real64(double, double)

declare void @binary_subtract_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float)

declare void @checked_binary_subtract_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float)

declare void @binary_subtract_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double)

declare void @checked_binary_subtract_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double)

declare i8 @binary_times_Integer8_Integer8(i8, i8)

declare i8 @checked_binary_times_Integer8_Integer8(i8, i8)

declare i16 @binary_times_Integer16_Integer16(i16, i16)

declare i16 @checked_binary_times_Integer16_Integer16(i16, i16)

declare i32 @binary_times_Integer32_Integer32(i32, i32)

declare i32 @checked_binary_times_Integer32_Integer32(i32, i32)

; Function Attrs: nounwind readnone speculatable
declare { i32, i1 } @llvm.smul.with.overflow.i32(i32, i32) #0

declare i64 @binary_times_Integer64_Integer64(i64, i64)

declare i64 @checked_binary_times_Integer64_Integer64(i64, i64)

declare i8 @binary_times_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i8 @checked_binary_times_UnsignedInteger8_UnsignedInteger8(i8, i8)

; Function Attrs: nounwind readnone speculatable
declare { i8, i1 } @llvm.umul.with.overflow.i8(i8, i8) #0

declare i16 @binary_times_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i16 @checked_binary_times_UnsignedInteger16_UnsignedInteger16(i16, i16)

; Function Attrs: nounwind readnone speculatable
declare { i16, i1 } @llvm.umul.with.overflow.i16(i16, i16) #0

declare i32 @binary_times_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i32 @checked_binary_times_UnsignedInteger32_UnsignedInteger32(i32, i32)

; Function Attrs: nounwind readnone speculatable
declare { i32, i1 } @llvm.umul.with.overflow.i32(i32, i32) #0

declare i64 @binary_times_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i64 @checked_binary_times_UnsignedInteger64_UnsignedInteger64(i64, i64)

; Function Attrs: nounwind readnone speculatable
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64) #0

declare i16 @binary_times_Real16_Real16(i16, i16)

declare i16 @checked_binary_times_Real16_Real16(i16, i16)

declare float @binary_times_Real32_Real32(float, float)

declare float @checked_binary_times_Real32_Real32(float, float)

declare double @binary_times_Real64_Real64(double, double)

declare double @checked_binary_times_Real64_Real64(double, double)

declare void @binary_times_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float)

declare void @checked_binary_times_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float)

declare void @binary_times_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double)

declare void @checked_binary_times_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double)

declare i8 @binary_unitize_Integer8_Integer8(i8, i8)

declare i8 @checked_binary_unitize_Integer8_Integer8(i8, i8)

declare i16 @binary_unitize_Integer16_Integer16(i16, i16)

declare i16 @checked_binary_unitize_Integer16_Integer16(i16, i16)

declare i32 @binary_unitize_Integer32_Integer32(i32, i32)

declare i32 @checked_binary_unitize_Integer32_Integer32(i32, i32)

declare i64 @binary_unitize_Integer64_Integer64(i64, i64)

declare i64 @checked_binary_unitize_Integer64_Integer64(i64, i64)

declare i8 @binary_unitize_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i8 @checked_binary_unitize_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i16 @binary_unitize_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i16 @checked_binary_unitize_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i32 @binary_unitize_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i32 @checked_binary_unitize_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i64 @binary_unitize_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i64 @checked_binary_unitize_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i64 @binary_unitize_Real16_Real16(i16, i16)

declare i32 @_ZN3wrtL7compareIdN10half_float4halfES2_EENSt3__19enable_ifIXaasr17is_floating_pointIT_EE5valueooaasr17is_floating_pointIT0_EE5valuentsr10is_complexIT1_EE5valueaasr17is_floating_pointIS7_EE5valuentsr10is_complexIS6_EE5valueEiE4typeERKS5_RKS6_RKS7_(i16, i16)

declare i32 @__cxa_guard_acquire(i64*)

declare void @__cxa_guard_release(i64*)

declare i64 @checked_binary_unitize_Real16_Real16(i16, i16)

declare i64 @binary_unitize_Real32_Real32(float, float)

declare i32 @_ZN3wrtL7compareIdffEENSt3__19enable_ifIXaasr17is_floating_pointIT_EE5valueooaasr17is_floating_pointIT0_EE5valuentsr10is_complexIT1_EE5valueaasr17is_floating_pointIS5_EE5valuentsr10is_complexIS4_EE5valueEiE4typeERKS3_RKS4_RKS5_(float, float)

declare i64 @checked_binary_unitize_Real32_Real32(float, float)

declare i64 @binary_unitize_Real64_Real64(double, double)

declare i32 @_ZN3wrtL7compareIdddEENSt3__19enable_ifIXaasr17is_floating_pointIT_EE5valueooaasr17is_floating_pointIT0_EE5valuentsr10is_complexIT1_EE5valueaasr17is_floating_pointIS5_EE5valuentsr10is_complexIS4_EE5valueEiE4typeERKS3_RKS4_RKS5_(double, double)

declare i64 @checked_binary_unitize_Real64_Real64(double, double)

declare i64 @binary_unitize_ComplexReal32_ComplexReal32(<2 x float>, <2 x float>)

declare i64 @checked_binary_unitize_ComplexReal32_ComplexReal32(<2 x float>, <2 x float>)

declare i64 @binary_unitize_ComplexReal64_ComplexReal64(double, double, double, double)

declare i64 @checked_binary_unitize_ComplexReal64_ComplexReal64(double, double, double, double)

declare i8 @binary_pow_Integer8_Integer8(i8, i8)

declare %"class.std::__1::basic_ostream"* @_ZNSt3__124__put_character_sequenceIcNS_11char_traitsIcEEEERNS_13basic_ostreamIT_T0_EES7_PKS4_m(%"class.std::__1::basic_ostream"*, i8*, i64)

declare void @_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE6sentryC1ERS3_(%"class.std::__1::basic_ostream<char, std::__1::char_traits<char> >::sentry"*, %"class.std::__1::basic_ostream"*)

declare void @_ZNKSt3__18ios_base6getlocEv(%"class.std::__1::locale"*, %"class.std::__1::ios_base"*)

declare %"class.std::__1::locale::facet"* @_ZNKSt3__16locale9use_facetERNS0_2idE(%"class.std::__1::locale"*, %"class.std::__1::locale::id"*)

declare void @_ZNSt3__16localeD1Ev(%"class.std::__1::locale"*)

declare %"class.std::__1::basic_streambuf"* @_ZNSt3__116__pad_and_outputIcNS_11char_traitsIcEEEENS_19ostreambuf_iteratorIT_T0_EES6_PKS4_S8_S8_RNS_8ios_baseES4_(%"class.std::__1::basic_streambuf"*, i8*, i8*, i8*, %"class.std::__1::ios_base"*, i8)

declare void @_ZNSt3__18ios_base5clearEj(%"class.std::__1::ios_base"*, i32)

declare void @_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE6sentryD1Ev(%"class.std::__1::basic_ostream<char, std::__1::char_traits<char> >::sentry"*)

declare void @_ZNSt3__18ios_base33__set_badbit_and_consider_rethrowEv(%"class.std::__1::ios_base"*)

declare i8* @_Znwm(i64)

declare i8 @checked_binary_pow_Integer8_Integer8(i8, i8)

declare i8 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIaaLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i8, i8)

declare %"class.std::__1::basic_ostream"* @_ZNSt3__1lsINS_11char_traitsIcEEEERNS_13basic_ostreamIcT_EES6_PKc(%"class.std::__1::basic_ostream"*, i8*)

declare i64 @strlen(i8*)

declare i8 @binary_pow_Integer8_UnsignedInteger8(i8, i8)

declare i8 @checked_binary_pow_Integer8_UnsignedInteger8(i8, i8)

declare i8 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIahLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i8, i8)

declare i8 @binary_pow_Integer8_Integer16(i8, i16)

declare i8 @checked_binary_pow_Integer8_Integer16(i8, i16)

declare i8 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIasLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i8, i16)

declare i8 @binary_pow_Integer8_UnsignedInteger16(i8, i16)

declare i8 @checked_binary_pow_Integer8_UnsignedInteger16(i8, i16)

declare i8 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIatLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i8, i16)

declare i8 @binary_pow_Integer8_Integer32(i8, i32)

declare i8 @checked_binary_pow_Integer8_Integer32(i8, i32)

declare i8 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIaiLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i8, i32)

declare i8 @binary_pow_Integer8_UnsignedInteger32(i8, i32)

declare i8 @checked_binary_pow_Integer8_UnsignedInteger32(i8, i32)

declare i8 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIajLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i8, i32)

declare i8 @binary_pow_Integer8_Integer64(i8, i64)

declare i8 @checked_binary_pow_Integer8_Integer64(i8, i64)

declare i8 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIaxLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i8, i64)

declare i8 @binary_pow_Integer8_UnsignedInteger64(i8, i64)

declare i8 @checked_binary_pow_Integer8_UnsignedInteger64(i8, i64)

declare i8 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIayLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i8, i64)

declare i16 @binary_pow_Integer8_Real16(i8, i16)

; Function Attrs: nounwind readnone speculatable
declare float @llvm.pow.f32(float, float) #0

declare i16 @checked_binary_pow_Integer8_Real16(i8, i16)

declare float @binary_pow_Integer8_Real32(i8, float)

declare float @checked_binary_pow_Integer8_Real32(i8, float)

declare void @binary_pow_Integer8_ComplexReal32(float*, float*, i8, float, float)

; Function Attrs: nounwind readnone speculatable
declare float @llvm.exp.f32(float) #0

; Function Attrs: nounwind readnone speculatable
declare float @llvm.cos.f32(float) #0

; Function Attrs: nounwind readnone speculatable
declare float @llvm.sin.f32(float) #0

declare void @checked_binary_pow_Integer8_ComplexReal32(float*, float*, i8, float, float)

declare double @binary_pow_Integer8_Real64(i8, double)

declare double @checked_binary_pow_Integer8_Real64(i8, double)

declare void @binary_pow_Integer8_ComplexReal64(double*, double*, i8, double, double)

declare void @checked_binary_pow_Integer8_ComplexReal64(double*, double*, i8, double, double)

declare i8 @binary_pow_UnsignedInteger8_Integer8(i8, i8)

declare i8 @checked_binary_pow_UnsignedInteger8_Integer8(i8, i8)

declare i8 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIhaLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i8, i8)

declare i8 @binary_pow_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i8 @checked_binary_pow_UnsignedInteger8_UnsignedInteger8(i8, i8)

declare i8 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIhhLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i8, i8)

declare i8 @binary_pow_UnsignedInteger8_Integer16(i8, i16)

declare i8 @checked_binary_pow_UnsignedInteger8_Integer16(i8, i16)

declare i8 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIhsLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i8, i16)

declare i8 @binary_pow_UnsignedInteger8_UnsignedInteger16(i8, i16)

declare i8 @checked_binary_pow_UnsignedInteger8_UnsignedInteger16(i8, i16)

declare i8 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIhtLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i8, i16)

declare i8 @binary_pow_UnsignedInteger8_Integer32(i8, i32)

declare i8 @checked_binary_pow_UnsignedInteger8_Integer32(i8, i32)

declare i8 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIhiLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i8, i32)

declare i8 @binary_pow_UnsignedInteger8_UnsignedInteger32(i8, i32)

declare i8 @checked_binary_pow_UnsignedInteger8_UnsignedInteger32(i8, i32)

declare i8 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIhjLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i8, i32)

declare i8 @binary_pow_UnsignedInteger8_Integer64(i8, i64)

declare i8 @checked_binary_pow_UnsignedInteger8_Integer64(i8, i64)

declare i8 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIhxLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i8, i64)

declare i8 @binary_pow_UnsignedInteger8_UnsignedInteger64(i8, i64)

declare i8 @checked_binary_pow_UnsignedInteger8_UnsignedInteger64(i8, i64)

declare i8 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIhyLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i8, i64)

declare i16 @binary_pow_UnsignedInteger8_Real16(i8, i16)

declare i16 @checked_binary_pow_UnsignedInteger8_Real16(i8, i16)

declare float @binary_pow_UnsignedInteger8_Real32(i8, float)

declare float @checked_binary_pow_UnsignedInteger8_Real32(i8, float)

declare void @binary_pow_UnsignedInteger8_ComplexReal32(float*, float*, i8, float, float)

declare void @checked_binary_pow_UnsignedInteger8_ComplexReal32(float*, float*, i8, float, float)

declare double @binary_pow_UnsignedInteger8_Real64(i8, double)

declare double @checked_binary_pow_UnsignedInteger8_Real64(i8, double)

declare void @binary_pow_UnsignedInteger8_ComplexReal64(double*, double*, i8, double, double)

declare void @checked_binary_pow_UnsignedInteger8_ComplexReal64(double*, double*, i8, double, double)

declare i16 @binary_pow_Integer16_Integer8(i16, i8)

declare i16 @checked_binary_pow_Integer16_Integer8(i16, i8)

declare i16 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIsaLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i16, i8)

declare i16 @binary_pow_Integer16_UnsignedInteger8(i16, i8)

declare i16 @checked_binary_pow_Integer16_UnsignedInteger8(i16, i8)

declare i16 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIshLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i16, i8)

declare i16 @binary_pow_Integer16_Integer16(i16, i16)

declare i16 @checked_binary_pow_Integer16_Integer16(i16, i16)

declare i16 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIssLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i16, i16)

declare i16 @binary_pow_Integer16_UnsignedInteger16(i16, i16)

declare i16 @checked_binary_pow_Integer16_UnsignedInteger16(i16, i16)

declare i16 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIstLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i16, i16)

declare i16 @binary_pow_Integer16_Integer32(i16, i32)

declare i16 @checked_binary_pow_Integer16_Integer32(i16, i32)

declare i16 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIsiLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i16, i32)

declare i16 @binary_pow_Integer16_UnsignedInteger32(i16, i32)

declare i16 @checked_binary_pow_Integer16_UnsignedInteger32(i16, i32)

declare i16 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIsjLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i16, i32)

declare i16 @binary_pow_Integer16_Integer64(i16, i64)

declare i16 @checked_binary_pow_Integer16_Integer64(i16, i64)

declare i16 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIsxLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i16, i64)

declare i16 @binary_pow_Integer16_UnsignedInteger64(i16, i64)

declare i16 @checked_binary_pow_Integer16_UnsignedInteger64(i16, i64)

declare i16 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIsyLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i16, i64)

declare i16 @binary_pow_Integer16_Real16(i16, i16)

declare i16 @checked_binary_pow_Integer16_Real16(i16, i16)

declare float @binary_pow_Integer16_Real32(i16, float)

declare float @checked_binary_pow_Integer16_Real32(i16, float)

declare void @binary_pow_Integer16_ComplexReal32(float*, float*, i16, float, float)

declare void @checked_binary_pow_Integer16_ComplexReal32(float*, float*, i16, float, float)

declare double @binary_pow_Integer16_Real64(i16, double)

declare double @checked_binary_pow_Integer16_Real64(i16, double)

declare void @binary_pow_Integer16_ComplexReal64(double*, double*, i16, double, double)

declare void @checked_binary_pow_Integer16_ComplexReal64(double*, double*, i16, double, double)

declare i16 @binary_pow_UnsignedInteger16_Integer8(i16, i8)

declare i16 @checked_binary_pow_UnsignedInteger16_Integer8(i16, i8)

declare i16 @_ZN3wrt9scalar_op6binaryL9pow_xi_opItaLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i16, i8)

declare i16 @binary_pow_UnsignedInteger16_UnsignedInteger8(i16, i8)

declare i16 @checked_binary_pow_UnsignedInteger16_UnsignedInteger8(i16, i8)

declare i16 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIthLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i16, i8)

declare i16 @binary_pow_UnsignedInteger16_Integer16(i16, i16)

declare i16 @checked_binary_pow_UnsignedInteger16_Integer16(i16, i16)

declare i16 @_ZN3wrt9scalar_op6binaryL9pow_xi_opItsLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i16, i16)

declare i16 @binary_pow_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i16 @checked_binary_pow_UnsignedInteger16_UnsignedInteger16(i16, i16)

declare i16 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIttLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i16, i16)

declare i16 @binary_pow_UnsignedInteger16_Integer32(i16, i32)

declare i16 @checked_binary_pow_UnsignedInteger16_Integer32(i16, i32)

declare i16 @_ZN3wrt9scalar_op6binaryL9pow_xi_opItiLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i16, i32)

declare i16 @binary_pow_UnsignedInteger16_UnsignedInteger32(i16, i32)

declare i16 @checked_binary_pow_UnsignedInteger16_UnsignedInteger32(i16, i32)

declare i16 @_ZN3wrt9scalar_op6binaryL9pow_xi_opItjLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i16, i32)

declare i16 @binary_pow_UnsignedInteger16_Integer64(i16, i64)

declare i16 @checked_binary_pow_UnsignedInteger16_Integer64(i16, i64)

declare i16 @_ZN3wrt9scalar_op6binaryL9pow_xi_opItxLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i16, i64)

declare i16 @binary_pow_UnsignedInteger16_UnsignedInteger64(i16, i64)

declare i16 @checked_binary_pow_UnsignedInteger16_UnsignedInteger64(i16, i64)

declare i16 @_ZN3wrt9scalar_op6binaryL9pow_xi_opItyLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i16, i64)

declare i16 @binary_pow_UnsignedInteger16_Real16(i16, i16)

declare i16 @checked_binary_pow_UnsignedInteger16_Real16(i16, i16)

declare float @binary_pow_UnsignedInteger16_Real32(i16, float)

declare float @checked_binary_pow_UnsignedInteger16_Real32(i16, float)

declare void @binary_pow_UnsignedInteger16_ComplexReal32(float*, float*, i16, float, float)

declare void @checked_binary_pow_UnsignedInteger16_ComplexReal32(float*, float*, i16, float, float)

declare double @binary_pow_UnsignedInteger16_Real64(i16, double)

declare double @checked_binary_pow_UnsignedInteger16_Real64(i16, double)

declare void @binary_pow_UnsignedInteger16_ComplexReal64(double*, double*, i16, double, double)

declare void @checked_binary_pow_UnsignedInteger16_ComplexReal64(double*, double*, i16, double, double)

declare i32 @binary_pow_Integer32_Integer8(i32, i8)

declare i32 @checked_binary_pow_Integer32_Integer8(i32, i8)

declare i32 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIiaLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i32, i8)

declare i32 @binary_pow_Integer32_UnsignedInteger8(i32, i8)

declare i32 @checked_binary_pow_Integer32_UnsignedInteger8(i32, i8)

declare i32 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIihLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i32, i8)

declare i32 @binary_pow_Integer32_Integer16(i32, i16)

declare i32 @checked_binary_pow_Integer32_Integer16(i32, i16)

declare i32 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIisLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i32, i16)

declare i32 @binary_pow_Integer32_UnsignedInteger16(i32, i16)

declare i32 @checked_binary_pow_Integer32_UnsignedInteger16(i32, i16)

declare i32 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIitLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i32, i16)

declare i32 @binary_pow_Integer32_Integer32(i32, i32)

declare i32 @checked_binary_pow_Integer32_Integer32(i32, i32)

declare i32 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIiiLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i32, i32)

declare i32 @binary_pow_Integer32_UnsignedInteger32(i32, i32)

declare i32 @checked_binary_pow_Integer32_UnsignedInteger32(i32, i32)

declare i32 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIijLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i32, i32)

declare i32 @binary_pow_Integer32_Integer64(i32, i64)

declare i32 @checked_binary_pow_Integer32_Integer64(i32, i64)

declare i32 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIixLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i32, i64)

declare i32 @binary_pow_Integer32_UnsignedInteger64(i32, i64)

declare i32 @checked_binary_pow_Integer32_UnsignedInteger64(i32, i64)

declare i32 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIiyLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i32, i64)

declare i16 @binary_pow_Integer32_Real16(i32, i16)

declare i16 @checked_binary_pow_Integer32_Real16(i32, i16)

declare float @binary_pow_Integer32_Real32(i32, float)

declare float @checked_binary_pow_Integer32_Real32(i32, float)

declare void @binary_pow_Integer32_ComplexReal32(float*, float*, i32, float, float)

declare void @checked_binary_pow_Integer32_ComplexReal32(float*, float*, i32, float, float)

declare double @binary_pow_Integer32_Real64(i32, double)

declare double @checked_binary_pow_Integer32_Real64(i32, double)

declare void @binary_pow_Integer32_ComplexReal64(double*, double*, i32, double, double)

declare void @checked_binary_pow_Integer32_ComplexReal64(double*, double*, i32, double, double)

declare i32 @binary_pow_UnsignedInteger32_Integer8(i32, i8)

declare i32 @checked_binary_pow_UnsignedInteger32_Integer8(i32, i8)

declare i32 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIjaLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i32, i8)

declare i32 @binary_pow_UnsignedInteger32_UnsignedInteger8(i32, i8)

declare i32 @checked_binary_pow_UnsignedInteger32_UnsignedInteger8(i32, i8)

declare i32 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIjhLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i32, i8)

declare i32 @binary_pow_UnsignedInteger32_Integer16(i32, i16)

declare i32 @checked_binary_pow_UnsignedInteger32_Integer16(i32, i16)

declare i32 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIjsLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i32, i16)

declare i32 @binary_pow_UnsignedInteger32_UnsignedInteger16(i32, i16)

declare i32 @checked_binary_pow_UnsignedInteger32_UnsignedInteger16(i32, i16)

declare i32 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIjtLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i32, i16)

declare i32 @binary_pow_UnsignedInteger32_Integer32(i32, i32)

declare i32 @checked_binary_pow_UnsignedInteger32_Integer32(i32, i32)

declare i32 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIjiLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i32, i32)

declare i32 @binary_pow_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i32 @checked_binary_pow_UnsignedInteger32_UnsignedInteger32(i32, i32)

declare i32 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIjjLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i32, i32)

declare i32 @binary_pow_UnsignedInteger32_Integer64(i32, i64)

declare i32 @checked_binary_pow_UnsignedInteger32_Integer64(i32, i64)

declare i32 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIjxLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i32, i64)

declare i32 @binary_pow_UnsignedInteger32_UnsignedInteger64(i32, i64)

declare i32 @checked_binary_pow_UnsignedInteger32_UnsignedInteger64(i32, i64)

declare i32 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIjyLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i32, i64)

declare i16 @binary_pow_UnsignedInteger32_Real16(i32, i16)

declare i16 @checked_binary_pow_UnsignedInteger32_Real16(i32, i16)

declare float @binary_pow_UnsignedInteger32_Real32(i32, float)

declare float @checked_binary_pow_UnsignedInteger32_Real32(i32, float)

declare void @binary_pow_UnsignedInteger32_ComplexReal32(float*, float*, i32, float, float)

declare void @checked_binary_pow_UnsignedInteger32_ComplexReal32(float*, float*, i32, float, float)

declare double @binary_pow_UnsignedInteger32_Real64(i32, double)

declare double @checked_binary_pow_UnsignedInteger32_Real64(i32, double)

declare void @binary_pow_UnsignedInteger32_ComplexReal64(double*, double*, i32, double, double)

declare void @checked_binary_pow_UnsignedInteger32_ComplexReal64(double*, double*, i32, double, double)

declare i64 @binary_pow_Integer64_Integer8(i64, i8)

declare i64 @checked_binary_pow_Integer64_Integer8(i64, i8)

declare i64 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIxaLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i64, i8)

declare i64 @binary_pow_Integer64_UnsignedInteger8(i64, i8)

declare i64 @checked_binary_pow_Integer64_UnsignedInteger8(i64, i8)

declare i64 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIxhLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i64, i8)

declare i64 @binary_pow_Integer64_Integer16(i64, i16)

declare i64 @checked_binary_pow_Integer64_Integer16(i64, i16)

declare i64 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIxsLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i64, i16)

declare i64 @binary_pow_Integer64_UnsignedInteger16(i64, i16)

declare i64 @checked_binary_pow_Integer64_UnsignedInteger16(i64, i16)

declare i64 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIxtLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i64, i16)

declare i64 @binary_pow_Integer64_Integer32(i64, i32)

declare i64 @checked_binary_pow_Integer64_Integer32(i64, i32)

declare i64 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIxiLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i64, i32)

declare i64 @binary_pow_Integer64_UnsignedInteger32(i64, i32)

declare i64 @checked_binary_pow_Integer64_UnsignedInteger32(i64, i32)

declare i64 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIxjLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i64, i32)

declare i64 @binary_pow_Integer64_Integer64(i64, i64)

declare i64 @checked_binary_pow_Integer64_Integer64(i64, i64)

declare i64 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIxxLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_.183(i64, i64)

declare i64 @binary_pow_Integer64_UnsignedInteger64(i64, i64)

declare i64 @checked_binary_pow_Integer64_UnsignedInteger64(i64, i64)

declare i64 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIxyLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i64, i64)

declare i16 @binary_pow_Integer64_Real16(i64, i16)

declare i16 @checked_binary_pow_Integer64_Real16(i64, i16)

declare float @binary_pow_Integer64_Real32(i64, float)

declare float @checked_binary_pow_Integer64_Real32(i64, float)

declare void @binary_pow_Integer64_ComplexReal32(float*, float*, i64, float, float)

declare void @checked_binary_pow_Integer64_ComplexReal32(float*, float*, i64, float, float)

declare double @binary_pow_Integer64_Real64(i64, double)

declare double @checked_binary_pow_Integer64_Real64(i64, double)

declare void @binary_pow_Integer64_ComplexReal64(double*, double*, i64, double, double)

declare void @checked_binary_pow_Integer64_ComplexReal64(double*, double*, i64, double, double)

declare i64 @binary_pow_UnsignedInteger64_Integer8(i64, i8)

declare i64 @checked_binary_pow_UnsignedInteger64_Integer8(i64, i8)

declare i64 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIyaLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i64, i8)

declare i64 @binary_pow_UnsignedInteger64_UnsignedInteger8(i64, i8)

declare i64 @checked_binary_pow_UnsignedInteger64_UnsignedInteger8(i64, i8)

declare i64 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIyhLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i64, i8)

declare i64 @binary_pow_UnsignedInteger64_Integer16(i64, i16)

declare i64 @checked_binary_pow_UnsignedInteger64_Integer16(i64, i16)

declare i64 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIysLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i64, i16)

declare i64 @binary_pow_UnsignedInteger64_UnsignedInteger16(i64, i16)

declare i64 @checked_binary_pow_UnsignedInteger64_UnsignedInteger16(i64, i16)

declare i64 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIytLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i64, i16)

declare i64 @binary_pow_UnsignedInteger64_Integer32(i64, i32)

declare i64 @checked_binary_pow_UnsignedInteger64_Integer32(i64, i32)

declare i64 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIyiLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i64, i32)

declare i64 @binary_pow_UnsignedInteger64_UnsignedInteger32(i64, i32)

declare i64 @checked_binary_pow_UnsignedInteger64_UnsignedInteger32(i64, i32)

declare i64 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIyjLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i64, i32)

declare i64 @binary_pow_UnsignedInteger64_Integer64(i64, i64)

declare i64 @checked_binary_pow_UnsignedInteger64_Integer64(i64, i64)

declare i64 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIyxLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i64, i64)

declare i64 @binary_pow_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i64 @checked_binary_pow_UnsignedInteger64_UnsignedInteger64(i64, i64)

declare i64 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIyyLNS_13runtime_flagsE1EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(i64, i64)

declare i16 @binary_pow_UnsignedInteger64_Real16(i64, i16)

declare i16 @checked_binary_pow_UnsignedInteger64_Real16(i64, i16)

declare float @binary_pow_UnsignedInteger64_Real32(i64, float)

declare float @checked_binary_pow_UnsignedInteger64_Real32(i64, float)

declare void @binary_pow_UnsignedInteger64_ComplexReal32(float*, float*, i64, float, float)

declare void @checked_binary_pow_UnsignedInteger64_ComplexReal32(float*, float*, i64, float, float)

declare double @binary_pow_UnsignedInteger64_Real64(i64, double)

declare double @checked_binary_pow_UnsignedInteger64_Real64(i64, double)

declare void @binary_pow_UnsignedInteger64_ComplexReal64(double*, double*, i64, double, double)

declare void @checked_binary_pow_UnsignedInteger64_ComplexReal64(double*, double*, i64, double, double)

declare i16 @binary_pow_Real16_Integer8(i16, i8)

declare i16 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIN10half_float4halfEaLNS_13runtime_flagsE0EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS9_RKS8_(i16, i8)

declare i16 @checked_binary_pow_Real16_Integer8(i16, i8)

declare i16 @binary_pow_Real16_UnsignedInteger8(i16, i8)

declare i16 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIN10half_float4halfEhLNS_13runtime_flagsE0EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS9_RKS8_(i16, i8)

declare i16 @checked_binary_pow_Real16_UnsignedInteger8(i16, i8)

declare i16 @binary_pow_Real16_Integer16(i16, i16)

declare i16 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIN10half_float4halfEsLNS_13runtime_flagsE0EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS9_RKS8_(i16, i16)

declare i16 @checked_binary_pow_Real16_Integer16(i16, i16)

declare i16 @binary_pow_Real16_UnsignedInteger16(i16, i16)

declare i16 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIN10half_float4halfEtLNS_13runtime_flagsE0EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS9_RKS8_(i16, i16)

declare i16 @checked_binary_pow_Real16_UnsignedInteger16(i16, i16)

declare i16 @binary_pow_Real16_Integer32(i16, i32)

declare i16 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIN10half_float4halfEiLNS_13runtime_flagsE0EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS9_RKS8_(i16, i32)

declare i16 @checked_binary_pow_Real16_Integer32(i16, i32)

declare i16 @binary_pow_Real16_UnsignedInteger32(i16, i32)

declare i16 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIN10half_float4halfEjLNS_13runtime_flagsE0EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS9_RKS8_(i16, i32)

declare i16 @checked_binary_pow_Real16_UnsignedInteger32(i16, i32)

declare i16 @binary_pow_Real16_Integer64(i16, i64)

declare i16 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIN10half_float4halfExLNS_13runtime_flagsE0EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS9_RKS8_(i16, i64)

declare i16 @checked_binary_pow_Real16_Integer64(i16, i64)

declare i16 @binary_pow_Real16_UnsignedInteger64(i16, i64)

declare i16 @_ZN3wrt9scalar_op6binaryL9pow_xi_opIN10half_float4halfEyLNS_13runtime_flagsE0EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS9_RKS8_(i16, i64)

declare i16 @checked_binary_pow_Real16_UnsignedInteger64(i16, i64)

declare i16 @binary_pow_Real16_Real16(i16, i16)

declare i16 @checked_binary_pow_Real16_Real16(i16, i16)

declare float @binary_pow_Real16_Real32(i16, float)

declare float @checked_binary_pow_Real16_Real32(i16, float)

declare void @binary_pow_Real16_ComplexReal32(float*, float*, i16, float, float)

declare void @checked_binary_pow_Real16_ComplexReal32(float*, float*, i16, float, float)

declare double @binary_pow_Real16_Real64(i16, double)

declare double @checked_binary_pow_Real16_Real64(i16, double)

declare void @binary_pow_Real16_ComplexReal64(double*, double*, i16, double, double)

declare void @checked_binary_pow_Real16_ComplexReal64(double*, double*, i16, double, double)

declare float @binary_pow_Real32_Integer8(float, i8)

declare float @_ZN3wrt9scalar_op6binaryL9pow_xi_opIfaLNS_13runtime_flagsE0EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(float, i8)

declare float @checked_binary_pow_Real32_Integer8(float, i8)

declare float @binary_pow_Real32_UnsignedInteger8(float, i8)

declare float @_ZN3wrt9scalar_op6binaryL9pow_xi_opIfhLNS_13runtime_flagsE0EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(float, i8)

declare float @checked_binary_pow_Real32_UnsignedInteger8(float, i8)

declare float @binary_pow_Real32_Integer16(float, i16)

declare float @_ZN3wrt9scalar_op6binaryL9pow_xi_opIfsLNS_13runtime_flagsE0EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(float, i16)

declare float @checked_binary_pow_Real32_Integer16(float, i16)

declare float @binary_pow_Real32_UnsignedInteger16(float, i16)

declare float @_ZN3wrt9scalar_op6binaryL9pow_xi_opIftLNS_13runtime_flagsE0EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(float, i16)

declare float @checked_binary_pow_Real32_UnsignedInteger16(float, i16)

declare float @binary_pow_Real32_Integer32(float, i32)

declare float @_ZN3wrt9scalar_op6binaryL9pow_xi_opIfiLNS_13runtime_flagsE0EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(float, i32)

declare float @checked_binary_pow_Real32_Integer32(float, i32)

declare float @binary_pow_Real32_UnsignedInteger32(float, i32)

declare float @_ZN3wrt9scalar_op6binaryL9pow_xi_opIfjLNS_13runtime_flagsE0EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(float, i32)

declare float @checked_binary_pow_Real32_UnsignedInteger32(float, i32)

declare float @binary_pow_Real32_Integer64(float, i64)

declare float @_ZN3wrt9scalar_op6binaryL9pow_xi_opIfxLNS_13runtime_flagsE0EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(float, i64)

declare float @checked_binary_pow_Real32_Integer64(float, i64)

declare float @binary_pow_Real32_UnsignedInteger64(float, i64)

declare float @_ZN3wrt9scalar_op6binaryL9pow_xi_opIfyLNS_13runtime_flagsE0EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(float, i64)

declare float @checked_binary_pow_Real32_UnsignedInteger64(float, i64)

declare float @binary_pow_Real32_Real16(float, i16)

declare float @checked_binary_pow_Real32_Real16(float, i16)

declare float @binary_pow_Real32_Real32(float, float)

declare float @checked_binary_pow_Real32_Real32(float, float)

declare void @binary_pow_Real32_ComplexReal32(float*, float*, float, float, float)

declare void @checked_binary_pow_Real32_ComplexReal32(float*, float*, float, float, float)

declare double @binary_pow_Real32_Real64(float, double)

declare double @checked_binary_pow_Real32_Real64(float, double)

declare void @binary_pow_Real32_ComplexReal64(double*, double*, float, double, double)

declare void @checked_binary_pow_Real32_ComplexReal64(double*, double*, float, double, double)

declare void @binary_pow_ComplexReal32_Integer8(float*, float*, float, float, i8)

declare <2 x float> @_ZN3wrt9scalar_op6binaryL9pow_xi_opINSt3__17complexIfEEaLNS_13runtime_flagsE0EEENS3_9enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS9_RKS8_(%"class.std::__1::complex.156"*, i8)

declare void @checked_binary_pow_ComplexReal32_Integer8(float*, float*, float, float, i8)

declare float @_ZN3wrt9scalar_op6binaryL6pow_opIfaLNS_13runtime_flagsE1EEENSt3__19enable_ifIXaasr17is_floating_pointIT_EE5valuesr11is_integralIT0_EE5valueENS1_6detail7op_infoILNS8_2opE20EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE(float*, i8)

declare void @binary_pow_ComplexReal32_UnsignedInteger8(float*, float*, float, float, i8)

declare <2 x float> @_ZN3wrt9scalar_op6binaryL9pow_xi_opINSt3__17complexIfEEhLNS_13runtime_flagsE0EEENS3_9enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS9_RKS8_(%"class.std::__1::complex.156"*, i8)

declare void @checked_binary_pow_ComplexReal32_UnsignedInteger8(float*, float*, float, float, i8)

declare void @binary_pow_ComplexReal32_Integer16(float*, float*, float, float, i16)

declare <2 x float> @_ZN3wrt9scalar_op6binaryL9pow_xi_opINSt3__17complexIfEEsLNS_13runtime_flagsE0EEENS3_9enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS9_RKS8_(%"class.std::__1::complex.156"*, i16)

declare void @checked_binary_pow_ComplexReal32_Integer16(float*, float*, float, float, i16)

declare float @_ZN3wrt9scalar_op6binaryL6pow_opIfsLNS_13runtime_flagsE1EEENSt3__19enable_ifIXaasr17is_floating_pointIT_EE5valuesr11is_integralIT0_EE5valueENS1_6detail7op_infoILNS8_2opE20EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE(float*, i16)

declare void @binary_pow_ComplexReal32_UnsignedInteger16(float*, float*, float, float, i16)

declare <2 x float> @_ZN3wrt9scalar_op6binaryL9pow_xi_opINSt3__17complexIfEEtLNS_13runtime_flagsE0EEENS3_9enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS9_RKS8_(%"class.std::__1::complex.156"*, i16)

declare void @checked_binary_pow_ComplexReal32_UnsignedInteger16(float*, float*, float, float, i16)

declare void @binary_pow_ComplexReal32_Integer32(float*, float*, float, float, i32)

declare <2 x float> @_ZN3wrt9scalar_op6binaryL9pow_xi_opINSt3__17complexIfEEiLNS_13runtime_flagsE0EEENS3_9enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS9_RKS8_(%"class.std::__1::complex.156"*, i32)

declare void @checked_binary_pow_ComplexReal32_Integer32(float*, float*, float, float, i32)

declare float @_ZN3wrt9scalar_op6binaryL6pow_opIfiLNS_13runtime_flagsE1EEENSt3__19enable_ifIXaasr17is_floating_pointIT_EE5valuesr11is_integralIT0_EE5valueENS1_6detail7op_infoILNS8_2opE20EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE(float*, i32)

declare void @binary_pow_ComplexReal32_UnsignedInteger32(float*, float*, float, float, i32)

declare <2 x float> @_ZN3wrt9scalar_op6binaryL9pow_xi_opINSt3__17complexIfEEjLNS_13runtime_flagsE0EEENS3_9enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS9_RKS8_(%"class.std::__1::complex.156"*, i32)

declare void @checked_binary_pow_ComplexReal32_UnsignedInteger32(float*, float*, float, float, i32)

declare void @binary_pow_ComplexReal32_Integer64(float*, float*, float, float, i64)

declare <2 x float> @_ZN3wrt9scalar_op6binaryL9pow_xi_opINSt3__17complexIfEExLNS_13runtime_flagsE0EEENS3_9enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS9_RKS8_(%"class.std::__1::complex.156"*, i64)

declare void @checked_binary_pow_ComplexReal32_Integer64(float*, float*, float, float, i64)

declare float @_ZN3wrt9scalar_op6binaryL6pow_opIfxLNS_13runtime_flagsE1EEENSt3__19enable_ifIXaasr17is_floating_pointIT_EE5valuesr11is_integralIT0_EE5valueENS1_6detail7op_infoILNS8_2opE20EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE(float*, i64)

declare void @binary_pow_ComplexReal32_UnsignedInteger64(float*, float*, float, float, i64)

declare <2 x float> @_ZN3wrt9scalar_op6binaryL9pow_xi_opINSt3__17complexIfEEyLNS_13runtime_flagsE0EEENS3_9enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS9_RKS8_(%"class.std::__1::complex.156"*, i64)

declare void @checked_binary_pow_ComplexReal32_UnsignedInteger64(float*, float*, float, float, i64)

declare void @binary_pow_ComplexReal32_Real16(float*, float*, float, float, i16)

declare void @checked_binary_pow_ComplexReal32_Real16(float*, float*, float, float, i16)

declare void @binary_pow_ComplexReal32_Real32(float*, float*, float, float, float)

declare void @checked_binary_pow_ComplexReal32_Real32(float*, float*, float, float, float)

declare void @binary_pow_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float)

declare void @checked_binary_pow_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float)

declare void @binary_pow_ComplexReal32_Real64(float*, float*, float, float, double)

declare void @checked_binary_pow_ComplexReal32_Real64(float*, float*, float, float, double)

declare void @binary_pow_ComplexReal32_ComplexReal64(double*, double*, float, float, double, double)

declare void @checked_binary_pow_ComplexReal32_ComplexReal64(double*, double*, float, float, double, double)

declare double @binary_pow_Real64_Integer8(double, i8)

declare double @_ZN3wrt9scalar_op6binaryL9pow_xi_opIdaLNS_13runtime_flagsE0EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(double, i8)

declare double @checked_binary_pow_Real64_Integer8(double, i8)

declare double @binary_pow_Real64_UnsignedInteger8(double, i8)

declare double @_ZN3wrt9scalar_op6binaryL9pow_xi_opIdhLNS_13runtime_flagsE0EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(double, i8)

declare double @checked_binary_pow_Real64_UnsignedInteger8(double, i8)

declare double @binary_pow_Real64_Integer16(double, i16)

declare double @_ZN3wrt9scalar_op6binaryL9pow_xi_opIdsLNS_13runtime_flagsE0EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(double, i16)

declare double @checked_binary_pow_Real64_Integer16(double, i16)

declare double @binary_pow_Real64_UnsignedInteger16(double, i16)

declare double @_ZN3wrt9scalar_op6binaryL9pow_xi_opIdtLNS_13runtime_flagsE0EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(double, i16)

declare double @checked_binary_pow_Real64_UnsignedInteger16(double, i16)

declare double @binary_pow_Real64_Integer32(double, i32)

declare double @_ZN3wrt9scalar_op6binaryL9pow_xi_opIdiLNS_13runtime_flagsE0EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(double, i32)

declare double @checked_binary_pow_Real64_Integer32(double, i32)

declare double @binary_pow_Real64_UnsignedInteger32(double, i32)

declare double @_ZN3wrt9scalar_op6binaryL9pow_xi_opIdjLNS_13runtime_flagsE0EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(double, i32)

declare double @checked_binary_pow_Real64_UnsignedInteger32(double, i32)

declare double @binary_pow_Real64_Integer64(double, i64)

declare double @_ZN3wrt9scalar_op6binaryL9pow_xi_opIdxLNS_13runtime_flagsE0EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_.184(double, i64)

declare double @checked_binary_pow_Real64_Integer64(double, i64)

declare double @binary_pow_Real64_UnsignedInteger64(double, i64)

declare double @_ZN3wrt9scalar_op6binaryL9pow_xi_opIdyLNS_13runtime_flagsE0EEENSt3__19enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS7_RKS6_(double, i64)

declare double @checked_binary_pow_Real64_UnsignedInteger64(double, i64)

declare double @binary_pow_Real64_Real16(double, i16)

declare double @checked_binary_pow_Real64_Real16(double, i16)

declare double @binary_pow_Real64_Real32(double, float)

declare double @checked_binary_pow_Real64_Real32(double, float)

declare void @binary_pow_Real64_ComplexReal32(float*, float*, double, float, float)

declare void @checked_binary_pow_Real64_ComplexReal32(float*, float*, double, float, float)

declare double @binary_pow_Real64_Real64(double, double)

declare double @checked_binary_pow_Real64_Real64(double, double)

declare void @binary_pow_Real64_ComplexReal64(double*, double*, double, double, double)

declare void @checked_binary_pow_Real64_ComplexReal64(double*, double*, double, double, double)

declare void @binary_pow_ComplexReal64_Integer8(double*, double*, double, double, i8)

declare { double, double } @_ZN3wrt9scalar_op6binaryL9pow_xi_opINSt3__17complexIdEEaLNS_13runtime_flagsE0EEENS3_9enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS9_RKS8_(%"class.std::__1::complex"*, i8)

declare void @checked_binary_pow_ComplexReal64_Integer8(double*, double*, double, double, i8)

declare double @_ZN3wrt9scalar_op6binaryL6pow_opIdaLNS_13runtime_flagsE1EEENSt3__19enable_ifIXaasr17is_floating_pointIT_EE5valuesr11is_integralIT0_EE5valueENS1_6detail7op_infoILNS8_2opE20EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE(double*, i8)

declare void @binary_pow_ComplexReal64_UnsignedInteger8(double*, double*, double, double, i8)

declare { double, double } @_ZN3wrt9scalar_op6binaryL9pow_xi_opINSt3__17complexIdEEhLNS_13runtime_flagsE0EEENS3_9enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS9_RKS8_(%"class.std::__1::complex"*, i8)

declare void @checked_binary_pow_ComplexReal64_UnsignedInteger8(double*, double*, double, double, i8)

declare void @binary_pow_ComplexReal64_Integer16(double*, double*, double, double, i16)

declare { double, double } @_ZN3wrt9scalar_op6binaryL9pow_xi_opINSt3__17complexIdEEsLNS_13runtime_flagsE0EEENS3_9enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS9_RKS8_(%"class.std::__1::complex"*, i16)

declare void @checked_binary_pow_ComplexReal64_Integer16(double*, double*, double, double, i16)

declare double @_ZN3wrt9scalar_op6binaryL6pow_opIdsLNS_13runtime_flagsE1EEENSt3__19enable_ifIXaasr17is_floating_pointIT_EE5valuesr11is_integralIT0_EE5valueENS1_6detail7op_infoILNS8_2opE20EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE(double*, i16)

declare void @binary_pow_ComplexReal64_UnsignedInteger16(double*, double*, double, double, i16)

declare { double, double } @_ZN3wrt9scalar_op6binaryL9pow_xi_opINSt3__17complexIdEEtLNS_13runtime_flagsE0EEENS3_9enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS9_RKS8_(%"class.std::__1::complex"*, i16)

declare void @checked_binary_pow_ComplexReal64_UnsignedInteger16(double*, double*, double, double, i16)

declare void @binary_pow_ComplexReal64_Integer32(double*, double*, double, double, i32)

declare { double, double } @_ZN3wrt9scalar_op6binaryL9pow_xi_opINSt3__17complexIdEEiLNS_13runtime_flagsE0EEENS3_9enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS9_RKS8_(%"class.std::__1::complex"*, i32)

declare void @checked_binary_pow_ComplexReal64_Integer32(double*, double*, double, double, i32)

declare double @_ZN3wrt9scalar_op6binaryL6pow_opIdiLNS_13runtime_flagsE1EEENSt3__19enable_ifIXaasr17is_floating_pointIT_EE5valuesr11is_integralIT0_EE5valueENS1_6detail7op_infoILNS8_2opE20EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE(double*, i32)

declare void @binary_pow_ComplexReal64_UnsignedInteger32(double*, double*, double, double, i32)

declare { double, double } @_ZN3wrt9scalar_op6binaryL9pow_xi_opINSt3__17complexIdEEjLNS_13runtime_flagsE0EEENS3_9enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS9_RKS8_(%"class.std::__1::complex"*, i32)

declare void @checked_binary_pow_ComplexReal64_UnsignedInteger32(double*, double*, double, double, i32)

declare void @binary_pow_ComplexReal64_Integer64(double*, double*, double, double, i64)

declare { double, double } @_ZN3wrt9scalar_op6binaryL9pow_xi_opINSt3__17complexIdEExLNS_13runtime_flagsE0EEENS3_9enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS9_RKS8_.185(%"class.std::__1::complex"*, i64)

declare void @checked_binary_pow_ComplexReal64_Integer64(double*, double*, double, double, i64)

declare double @_ZN3wrt9scalar_op6binaryL6pow_opIdxLNS_13runtime_flagsE1EEENSt3__19enable_ifIXaasr17is_floating_pointIT_EE5valuesr11is_integralIT0_EE5valueENS1_6detail7op_infoILNS8_2opE20EN7rt_typeIS6_E4typeENSB_IS7_E4typeEE11result_typeEE4typeERKNSG_19first_argument_typeERKNSG_20second_argument_typeE.186(double*, i64)

declare void @binary_pow_ComplexReal64_UnsignedInteger64(double*, double*, double, double, i64)

declare { double, double } @_ZN3wrt9scalar_op6binaryL9pow_xi_opINSt3__17complexIdEEyLNS_13runtime_flagsE0EEENS3_9enable_ifIXsr11is_integralIT0_EE5valueET_E4typeERKS9_RKS8_(%"class.std::__1::complex"*, i64)

declare void @checked_binary_pow_ComplexReal64_UnsignedInteger64(double*, double*, double, double, i64)

declare void @binary_pow_ComplexReal64_Real16(double*, double*, double, double, i16)

declare void @checked_binary_pow_ComplexReal64_Real16(double*, double*, double, double, i16)

declare void @binary_pow_ComplexReal64_Real32(double*, double*, double, double, float)

declare void @checked_binary_pow_ComplexReal64_Real32(double*, double*, double, double, float)

declare void @binary_pow_ComplexReal64_ComplexReal32(double*, double*, double, double, float, float)

declare void @checked_binary_pow_ComplexReal64_ComplexReal32(double*, double*, double, double, float, float)

declare void @binary_pow_ComplexReal64_Real64(double*, double*, double, double, double)

declare void @checked_binary_pow_ComplexReal64_Real64(double*, double*, double, double, double)

declare void @binary_pow_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double)

declare void @checked_binary_pow_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double)

declare i8 @integer_safe_cast_Integer8_Integer8(i8)

declare i8 @integer_safe_cast_Integer8_UnsignedInteger8(i8)

declare i16 @integer_safe_cast_Integer8_Integer16(i8)

declare i16 @integer_safe_cast_Integer8_UnsignedInteger16(i8)

declare i32 @integer_safe_cast_Integer8_Integer32(i8)

declare i32 @integer_safe_cast_Integer8_UnsignedInteger32(i8)

declare i64 @integer_safe_cast_Integer8_Integer64(i8)

declare i64 @integer_safe_cast_Integer8_UnsignedInteger64(i8)

declare i8 @integer_safe_cast_UnsignedInteger8_Integer8(i8)

declare i8 @integer_safe_cast_UnsignedInteger8_UnsignedInteger8(i8)

declare i16 @integer_safe_cast_UnsignedInteger8_Integer16(i8)

declare i16 @integer_safe_cast_UnsignedInteger8_UnsignedInteger16(i8)

declare i32 @integer_safe_cast_UnsignedInteger8_Integer32(i8)

declare i32 @integer_safe_cast_UnsignedInteger8_UnsignedInteger32(i8)

declare i64 @integer_safe_cast_UnsignedInteger8_Integer64(i8)

declare i64 @integer_safe_cast_UnsignedInteger8_UnsignedInteger64(i8)

declare i8 @integer_safe_cast_Integer16_Integer8(i16)

declare i8 @integer_safe_cast_Integer16_UnsignedInteger8(i16)

declare i16 @integer_safe_cast_Integer16_Integer16(i16)

declare i16 @integer_safe_cast_Integer16_UnsignedInteger16(i16)

declare i32 @integer_safe_cast_Integer16_Integer32(i16)

declare i32 @integer_safe_cast_Integer16_UnsignedInteger32(i16)

declare i64 @integer_safe_cast_Integer16_Integer64(i16)

declare i64 @integer_safe_cast_Integer16_UnsignedInteger64(i16)

declare i8 @integer_safe_cast_UnsignedInteger16_Integer8(i16)

declare i8 @integer_safe_cast_UnsignedInteger16_UnsignedInteger8(i16)

declare i16 @integer_safe_cast_UnsignedInteger16_Integer16(i16)

declare i16 @integer_safe_cast_UnsignedInteger16_UnsignedInteger16(i16)

declare i32 @integer_safe_cast_UnsignedInteger16_Integer32(i16)

declare i32 @integer_safe_cast_UnsignedInteger16_UnsignedInteger32(i16)

declare i64 @integer_safe_cast_UnsignedInteger16_Integer64(i16)

declare i64 @integer_safe_cast_UnsignedInteger16_UnsignedInteger64(i16)

declare i8 @integer_safe_cast_Integer32_Integer8(i32)

declare i8 @integer_safe_cast_Integer32_UnsignedInteger8(i32)

declare i16 @integer_safe_cast_Integer32_Integer16(i32)

declare i16 @integer_safe_cast_Integer32_UnsignedInteger16(i32)

declare i32 @integer_safe_cast_Integer32_Integer32(i32)

declare i32 @integer_safe_cast_Integer32_UnsignedInteger32(i32)

declare i64 @integer_safe_cast_Integer32_Integer64(i32)

declare i64 @integer_safe_cast_Integer32_UnsignedInteger64(i32)

declare i8 @integer_safe_cast_UnsignedInteger32_Integer8(i32)

declare i8 @integer_safe_cast_UnsignedInteger32_UnsignedInteger8(i32)

declare i16 @integer_safe_cast_UnsignedInteger32_Integer16(i32)

declare i16 @integer_safe_cast_UnsignedInteger32_UnsignedInteger16(i32)

declare i32 @integer_safe_cast_UnsignedInteger32_Integer32(i32)

declare i32 @integer_safe_cast_UnsignedInteger32_UnsignedInteger32(i32)

declare i64 @integer_safe_cast_UnsignedInteger32_Integer64(i32)

declare i64 @integer_safe_cast_UnsignedInteger32_UnsignedInteger64(i32)

declare i8 @integer_safe_cast_Integer64_Integer8(i64)

declare i8 @integer_safe_cast_Integer64_UnsignedInteger8(i64)

declare i16 @integer_safe_cast_Integer64_Integer16(i64)

declare i16 @integer_safe_cast_Integer64_UnsignedInteger16(i64)

declare i32 @integer_safe_cast_Integer64_Integer32(i64)

declare i32 @integer_safe_cast_Integer64_UnsignedInteger32(i64)

declare i64 @integer_safe_cast_Integer64_Integer64(i64)

declare i64 @integer_safe_cast_Integer64_UnsignedInteger64(i64)

declare i8 @integer_safe_cast_UnsignedInteger64_Integer8(i64)

declare i8 @integer_safe_cast_UnsignedInteger64_UnsignedInteger8(i64)

declare i16 @integer_safe_cast_UnsignedInteger64_Integer16(i64)

declare i16 @integer_safe_cast_UnsignedInteger64_UnsignedInteger16(i64)

declare i32 @integer_safe_cast_UnsignedInteger64_Integer32(i64)

declare i32 @integer_safe_cast_UnsignedInteger64_UnsignedInteger32(i64)

declare i64 @integer_safe_cast_UnsignedInteger64_Integer64(i64)

declare i64 @integer_safe_cast_UnsignedInteger64_UnsignedInteger64(i64)

declare i8 @ternary_adxmy_Integer8_Integer8_Integer8(i8, i8, i8)

declare i8 @checked_ternary_adxmy_Integer8_Integer8_Integer8(i8, i8, i8)

declare i16 @ternary_adxmy_Integer16_Integer16_Integer16(i16, i16, i16)

declare i16 @checked_ternary_adxmy_Integer16_Integer16_Integer16(i16, i16, i16)

declare i32 @ternary_adxmy_Integer32_Integer32_Integer32(i32, i32, i32)

declare i32 @checked_ternary_adxmy_Integer32_Integer32_Integer32(i32, i32, i32)

declare i64 @ternary_adxmy_Integer64_Integer64_Integer64(i64, i64, i64)

declare i64 @checked_ternary_adxmy_Integer64_Integer64_Integer64(i64, i64, i64)

declare i8 @ternary_adxmy_UnsignedInteger8_UnsignedInteger8_UnsignedInteger8(i8, i8, i8)

declare i8 @checked_ternary_adxmy_UnsignedInteger8_UnsignedInteger8_UnsignedInteger8(i8, i8, i8)

declare i16 @ternary_adxmy_UnsignedInteger16_UnsignedInteger16_UnsignedInteger16(i16, i16, i16)

declare i16 @checked_ternary_adxmy_UnsignedInteger16_UnsignedInteger16_UnsignedInteger16(i16, i16, i16)

declare i32 @ternary_adxmy_UnsignedInteger32_UnsignedInteger32_UnsignedInteger32(i32, i32, i32)

declare i32 @checked_ternary_adxmy_UnsignedInteger32_UnsignedInteger32_UnsignedInteger32(i32, i32, i32)

declare i64 @ternary_adxmy_UnsignedInteger64_UnsignedInteger64_UnsignedInteger64(i64, i64, i64)

declare i64 @checked_ternary_adxmy_UnsignedInteger64_UnsignedInteger64_UnsignedInteger64(i64, i64, i64)

declare i16 @ternary_adxmy_Real16_Real16_Real16(i16, i16, i16)

declare i16 @checked_ternary_adxmy_Real16_Real16_Real16(i16, i16, i16)

declare float @ternary_adxmy_Real32_Real32_Real32(float, float, float)

declare float @checked_ternary_adxmy_Real32_Real32_Real32(float, float, float)

declare double @ternary_adxmy_Real64_Real64_Real64(double, double, double)

declare double @checked_ternary_adxmy_Real64_Real64_Real64(double, double, double)

declare void @ternary_adxmy_ComplexReal32_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float, float, float)

declare void @checked_ternary_adxmy_ComplexReal32_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float, float, float)

declare void @ternary_adxmy_ComplexReal64_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double, double, double)

declare void @checked_ternary_adxmy_ComplexReal64_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double, double, double)

declare i8 @ternary_axmy_Integer8_Integer8_Integer8(i8, i8, i8)

declare i8 @checked_ternary_axmy_Integer8_Integer8_Integer8(i8, i8, i8)

declare i16 @ternary_axmy_Integer16_Integer16_Integer16(i16, i16, i16)

declare i16 @checked_ternary_axmy_Integer16_Integer16_Integer16(i16, i16, i16)

declare i32 @ternary_axmy_Integer32_Integer32_Integer32(i32, i32, i32)

declare i32 @checked_ternary_axmy_Integer32_Integer32_Integer32(i32, i32, i32)

declare i64 @ternary_axmy_Integer64_Integer64_Integer64(i64, i64, i64)

declare i64 @checked_ternary_axmy_Integer64_Integer64_Integer64(i64, i64, i64)

declare i8 @ternary_axmy_UnsignedInteger8_UnsignedInteger8_UnsignedInteger8(i8, i8, i8)

declare i8 @checked_ternary_axmy_UnsignedInteger8_UnsignedInteger8_UnsignedInteger8(i8, i8, i8)

declare i16 @ternary_axmy_UnsignedInteger16_UnsignedInteger16_UnsignedInteger16(i16, i16, i16)

declare i16 @checked_ternary_axmy_UnsignedInteger16_UnsignedInteger16_UnsignedInteger16(i16, i16, i16)

declare i32 @ternary_axmy_UnsignedInteger32_UnsignedInteger32_UnsignedInteger32(i32, i32, i32)

declare i32 @checked_ternary_axmy_UnsignedInteger32_UnsignedInteger32_UnsignedInteger32(i32, i32, i32)

declare i64 @ternary_axmy_UnsignedInteger64_UnsignedInteger64_UnsignedInteger64(i64, i64, i64)

declare i64 @checked_ternary_axmy_UnsignedInteger64_UnsignedInteger64_UnsignedInteger64(i64, i64, i64)

declare i16 @ternary_axmy_Real16_Real16_Real16(i16, i16, i16)

declare i16 @checked_ternary_axmy_Real16_Real16_Real16(i16, i16, i16)

declare float @ternary_axmy_Real32_Real32_Real32(float, float, float)

declare float @checked_ternary_axmy_Real32_Real32_Real32(float, float, float)

declare double @ternary_axmy_Real64_Real64_Real64(double, double, double)

declare double @checked_ternary_axmy_Real64_Real64_Real64(double, double, double)

declare void @ternary_axmy_ComplexReal32_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float, float, float)

declare void @checked_ternary_axmy_ComplexReal32_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float, float, float)

declare void @ternary_axmy_ComplexReal64_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double, double, double)

declare void @checked_ternary_axmy_ComplexReal64_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double, double, double)

declare i8 @ternary_axpy_Integer8_Integer8_Integer8(i8, i8, i8)

declare i8 @checked_ternary_axpy_Integer8_Integer8_Integer8(i8, i8, i8)

declare i16 @ternary_axpy_Integer16_Integer16_Integer16(i16, i16, i16)

declare i16 @checked_ternary_axpy_Integer16_Integer16_Integer16(i16, i16, i16)

declare i32 @ternary_axpy_Integer32_Integer32_Integer32(i32, i32, i32)

declare i32 @checked_ternary_axpy_Integer32_Integer32_Integer32(i32, i32, i32)

declare i64 @ternary_axpy_Integer64_Integer64_Integer64(i64, i64, i64)

declare i64 @checked_ternary_axpy_Integer64_Integer64_Integer64(i64, i64, i64)

declare i8 @ternary_axpy_UnsignedInteger8_UnsignedInteger8_UnsignedInteger8(i8, i8, i8)

declare i8 @checked_ternary_axpy_UnsignedInteger8_UnsignedInteger8_UnsignedInteger8(i8, i8, i8)

declare i16 @ternary_axpy_UnsignedInteger16_UnsignedInteger16_UnsignedInteger16(i16, i16, i16)

declare i16 @checked_ternary_axpy_UnsignedInteger16_UnsignedInteger16_UnsignedInteger16(i16, i16, i16)

declare i32 @ternary_axpy_UnsignedInteger32_UnsignedInteger32_UnsignedInteger32(i32, i32, i32)

declare i32 @checked_ternary_axpy_UnsignedInteger32_UnsignedInteger32_UnsignedInteger32(i32, i32, i32)

declare i64 @ternary_axpy_UnsignedInteger64_UnsignedInteger64_UnsignedInteger64(i64, i64, i64)

declare i64 @checked_ternary_axpy_UnsignedInteger64_UnsignedInteger64_UnsignedInteger64(i64, i64, i64)

declare i16 @ternary_axpy_Real16_Real16_Real16(i16, i16, i16)

declare i16 @checked_ternary_axpy_Real16_Real16_Real16(i16, i16, i16)

declare float @ternary_axpy_Real32_Real32_Real32(float, float, float)

declare float @checked_ternary_axpy_Real32_Real32_Real32(float, float, float)

declare double @ternary_axpy_Real64_Real64_Real64(double, double, double)

declare double @checked_ternary_axpy_Real64_Real64_Real64(double, double, double)

declare void @ternary_axpy_ComplexReal32_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float, float, float)

declare void @checked_ternary_axpy_ComplexReal32_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float, float, float)

declare void @ternary_axpy_ComplexReal64_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double, double, double)

declare void @checked_ternary_axpy_ComplexReal64_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double, double, double)

declare i8 @ternary_ymax_Integer8_Integer8_Integer8(i8, i8, i8)

declare i8 @checked_ternary_ymax_Integer8_Integer8_Integer8(i8, i8, i8)

declare i16 @ternary_ymax_Integer16_Integer16_Integer16(i16, i16, i16)

declare i16 @checked_ternary_ymax_Integer16_Integer16_Integer16(i16, i16, i16)

declare i32 @ternary_ymax_Integer32_Integer32_Integer32(i32, i32, i32)

declare i32 @checked_ternary_ymax_Integer32_Integer32_Integer32(i32, i32, i32)

declare i64 @ternary_ymax_Integer64_Integer64_Integer64(i64, i64, i64)

declare i64 @checked_ternary_ymax_Integer64_Integer64_Integer64(i64, i64, i64)

declare i8 @ternary_ymax_UnsignedInteger8_UnsignedInteger8_UnsignedInteger8(i8, i8, i8)

declare i8 @checked_ternary_ymax_UnsignedInteger8_UnsignedInteger8_UnsignedInteger8(i8, i8, i8)

declare i16 @ternary_ymax_UnsignedInteger16_UnsignedInteger16_UnsignedInteger16(i16, i16, i16)

declare i16 @checked_ternary_ymax_UnsignedInteger16_UnsignedInteger16_UnsignedInteger16(i16, i16, i16)

declare i32 @ternary_ymax_UnsignedInteger32_UnsignedInteger32_UnsignedInteger32(i32, i32, i32)

declare i32 @checked_ternary_ymax_UnsignedInteger32_UnsignedInteger32_UnsignedInteger32(i32, i32, i32)

declare i64 @ternary_ymax_UnsignedInteger64_UnsignedInteger64_UnsignedInteger64(i64, i64, i64)

declare i64 @checked_ternary_ymax_UnsignedInteger64_UnsignedInteger64_UnsignedInteger64(i64, i64, i64)

declare i16 @ternary_ymax_Real16_Real16_Real16(i16, i16, i16)

declare i16 @checked_ternary_ymax_Real16_Real16_Real16(i16, i16, i16)

declare float @ternary_ymax_Real32_Real32_Real32(float, float, float)

declare float @checked_ternary_ymax_Real32_Real32_Real32(float, float, float)

declare double @ternary_ymax_Real64_Real64_Real64(double, double, double)

declare double @checked_ternary_ymax_Real64_Real64_Real64(double, double, double)

declare void @ternary_ymax_ComplexReal32_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float, float, float)

declare void @checked_ternary_ymax_ComplexReal32_ComplexReal32_ComplexReal32(float*, float*, float, float, float, float, float, float)

declare void @ternary_ymax_ComplexReal64_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double, double, double)

declare void @checked_ternary_ymax_ComplexReal64_ComplexReal64_ComplexReal64(double*, double*, double, double, double, double, double, double)

declare i8 @unary_abs_Integer8(i8)

declare i8 @checked_unary_abs_Integer8(i8)

declare i16 @unary_abs_Integer16(i16)

declare i16 @checked_unary_abs_Integer16(i16)

declare i32 @unary_abs_Integer32(i32)

declare i32 @checked_unary_abs_Integer32(i32)

declare i64 @unary_abs_Integer64(i64)

declare i64 @checked_unary_abs_Integer64(i64)

declare i8 @unary_abs_UnsignedInteger8(i8)

declare i8 @checked_unary_abs_UnsignedInteger8(i8)

declare i16 @unary_abs_UnsignedInteger16(i16)

declare i16 @checked_unary_abs_UnsignedInteger16(i16)

declare i32 @unary_abs_UnsignedInteger32(i32)

declare i32 @checked_unary_abs_UnsignedInteger32(i32)

declare i64 @unary_abs_UnsignedInteger64(i64)

declare i64 @checked_unary_abs_UnsignedInteger64(i64)

declare i16 @unary_abs_Real16(i16)

declare i16 @checked_unary_abs_Real16(i16)

declare float @unary_abs_Real32(float)

declare float @checked_unary_abs_Real32(float)

declare double @unary_abs_Real64(double)

declare double @checked_unary_abs_Real64(double)

declare float @unary_abs_ComplexReal32(float, float)

declare float @checked_unary_abs_ComplexReal32(float, float)

declare double @unary_abs_ComplexReal64(double, double)

declare double @checked_unary_abs_ComplexReal64(double, double)

declare i8 @unary_abssquare_Integer8(i8)

declare i8 @checked_unary_abssquare_Integer8(i8)

declare i16 @unary_abssquare_Integer16(i16)

declare i16 @checked_unary_abssquare_Integer16(i16)

declare i32 @unary_abssquare_Integer32(i32)

declare i32 @checked_unary_abssquare_Integer32(i32)

declare i64 @unary_abssquare_Integer64(i64)

declare i64 @checked_unary_abssquare_Integer64(i64)

declare i8 @unary_abssquare_UnsignedInteger8(i8)

declare i8 @checked_unary_abssquare_UnsignedInteger8(i8)

declare i16 @unary_abssquare_UnsignedInteger16(i16)

declare i16 @checked_unary_abssquare_UnsignedInteger16(i16)

declare i32 @unary_abssquare_UnsignedInteger32(i32)

declare i32 @checked_unary_abssquare_UnsignedInteger32(i32)

declare i64 @unary_abssquare_UnsignedInteger64(i64)

declare i64 @checked_unary_abssquare_UnsignedInteger64(i64)

declare i16 @unary_abssquare_Real16(i16)

declare i16 @checked_unary_abssquare_Real16(i16)

declare float @unary_abssquare_Real32(float)

declare float @checked_unary_abssquare_Real32(float)

declare double @unary_abssquare_Real64(double)

declare double @checked_unary_abssquare_Real64(double)

declare float @unary_abssquare_ComplexReal32(<2 x float>)

declare float @checked_unary_abssquare_ComplexReal32(<2 x float>)

declare double @unary_abssquare_ComplexReal64(double, double)

declare double @checked_unary_abssquare_ComplexReal64(double, double)

declare i8 @unary_acos_Integer8(i8)

declare i8 @checked_unary_acos_Integer8(i8)

declare i16 @unary_acos_Integer16(i16)

declare i16 @checked_unary_acos_Integer16(i16)

declare i32 @unary_acos_Integer32(i32)

declare i32 @checked_unary_acos_Integer32(i32)

declare i64 @unary_acos_Integer64(i64)

declare i64 @checked_unary_acos_Integer64(i64)

declare i8 @unary_acos_UnsignedInteger8(i8)

declare i8 @checked_unary_acos_UnsignedInteger8(i8)

declare i16 @unary_acos_UnsignedInteger16(i16)

declare i16 @checked_unary_acos_UnsignedInteger16(i16)

declare i32 @unary_acos_UnsignedInteger32(i32)

declare i32 @checked_unary_acos_UnsignedInteger32(i32)

declare i64 @unary_acos_UnsignedInteger64(i64)

declare i64 @checked_unary_acos_UnsignedInteger64(i64)

declare i16 @unary_acos_Real16(i16)

declare float @acosf(float)

declare i16 @checked_unary_acos_Real16(i16)

declare float @unary_acos_Real32(float)

declare float @checked_unary_acos_Real32(float)

declare double @unary_acos_Real64(double)

declare double @checked_unary_acos_Real64(double)

declare void @unary_acos_ComplexReal32(float*, float*, float, float)

declare <2 x float> @_ZN3wrt9scalar_op5unaryL8asinh_opINSt3__17complexIfEELNS_13runtime_flagsE0EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE13EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE(%"class.std::__1::complex.156"*)

declare float @frexpf(float, i32*)

declare float @log1pf(float)

declare void @checked_unary_acos_ComplexReal32(float*, float*, float, float)

declare void @unary_acos_ComplexReal64(double*, double*, double, double)

declare { double, double } @_ZN3wrt9scalar_op5unaryL8asinh_opINSt3__17complexIdEELNS_13runtime_flagsE0EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE13EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE.203(%"class.std::__1::complex"*)

declare { double, double } @_ZN3wrt9scalar_op5unaryL8log1p_opINSt3__17complexIdEELNS_13runtime_flagsE0EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE47EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE.204(double, double)

declare void @checked_unary_acos_ComplexReal64(double*, double*, double, double)

declare i8 @unary_acosh_Integer8(i8)

declare i8 @checked_unary_acosh_Integer8(i8)

declare i16 @unary_acosh_Integer16(i16)

declare i16 @checked_unary_acosh_Integer16(i16)

declare i32 @unary_acosh_Integer32(i32)

declare i32 @checked_unary_acosh_Integer32(i32)

declare i64 @unary_acosh_Integer64(i64)

declare i64 @checked_unary_acosh_Integer64(i64)

declare i8 @unary_acosh_UnsignedInteger8(i8)

declare i8 @checked_unary_acosh_UnsignedInteger8(i8)

declare i16 @unary_acosh_UnsignedInteger16(i16)

declare i16 @checked_unary_acosh_UnsignedInteger16(i16)

declare i32 @unary_acosh_UnsignedInteger32(i32)

declare i32 @checked_unary_acosh_UnsignedInteger32(i32)

declare i64 @unary_acosh_UnsignedInteger64(i64)

declare i64 @checked_unary_acosh_UnsignedInteger64(i64)

declare i16 @unary_acosh_Real16(i16)

declare float @acoshf(float)

declare i16 @checked_unary_acosh_Real16(i16)

declare float @unary_acosh_Real32(float)

declare float @checked_unary_acosh_Real32(float)

declare double @unary_acosh_Real64(double)

declare double @checked_unary_acosh_Real64(double)

declare void @unary_acosh_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_acosh_ComplexReal32(float*, float*, float, float)

declare void @unary_acosh_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_acosh_ComplexReal64(double*, double*, double, double)

declare i8 @unary_acot_Integer8(i8)

declare i8 @checked_unary_acot_Integer8(i8)

declare i16 @unary_acot_Integer16(i16)

declare i16 @checked_unary_acot_Integer16(i16)

declare i32 @unary_acot_Integer32(i32)

declare i32 @checked_unary_acot_Integer32(i32)

declare i64 @unary_acot_Integer64(i64)

declare i64 @checked_unary_acot_Integer64(i64)

declare i8 @unary_acot_UnsignedInteger8(i8)

declare i8 @checked_unary_acot_UnsignedInteger8(i8)

declare i16 @unary_acot_UnsignedInteger16(i16)

declare i16 @checked_unary_acot_UnsignedInteger16(i16)

declare i32 @unary_acot_UnsignedInteger32(i32)

declare i32 @checked_unary_acot_UnsignedInteger32(i32)

declare i64 @unary_acot_UnsignedInteger64(i64)

declare i64 @checked_unary_acot_UnsignedInteger64(i64)

declare i16 @unary_acot_Real16(i16)

declare float @atanf(float)

declare i16 @checked_unary_acot_Real16(i16)

declare float @unary_acot_Real32(float)

declare float @checked_unary_acot_Real32(float)

declare double @unary_acot_Real64(double)

declare double @checked_unary_acot_Real64(double)

declare void @unary_acot_ComplexReal32(float*, float*, float, float)

declare <2 x float> @_ZN3wrt9scalar_op5unaryL8atanh_opINSt3__17complexIfEELNS_13runtime_flagsE0EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE15EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE(%"class.std::__1::complex.156"*)

declare <2 x float> @_ZN3wrt9scalar_op5unaryL8rsqrt_opINSt3__17complexIfEELNS_13runtime_flagsE0EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE64EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE.206(%"class.std::__1::complex.156"*)

declare void @checked_unary_acot_ComplexReal32(float*, float*, float, float)

declare void @unary_acot_ComplexReal64(double*, double*, double, double)

declare { double, double } @_ZN3wrt9scalar_op5unaryL8atanh_opINSt3__17complexIdEELNS_13runtime_flagsE0EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE15EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE.210(%"class.std::__1::complex"*)

declare { double, double } @_ZN3wrt9scalar_op5unaryL8rsqrt_opINSt3__17complexIdEELNS_13runtime_flagsE0EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE64EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE.211(%"class.std::__1::complex"*)

declare void @checked_unary_acot_ComplexReal64(double*, double*, double, double)

declare i8 @unary_acoth_Integer8(i8)

declare i8 @checked_unary_acoth_Integer8(i8)

declare i16 @unary_acoth_Integer16(i16)

declare i16 @checked_unary_acoth_Integer16(i16)

declare i32 @unary_acoth_Integer32(i32)

declare i32 @checked_unary_acoth_Integer32(i32)

declare i64 @unary_acoth_Integer64(i64)

declare i64 @checked_unary_acoth_Integer64(i64)

declare i8 @unary_acoth_UnsignedInteger8(i8)

declare i8 @checked_unary_acoth_UnsignedInteger8(i8)

declare i16 @unary_acoth_UnsignedInteger16(i16)

declare i16 @checked_unary_acoth_UnsignedInteger16(i16)

declare i32 @unary_acoth_UnsignedInteger32(i32)

declare i32 @checked_unary_acoth_UnsignedInteger32(i32)

declare i64 @unary_acoth_UnsignedInteger64(i64)

declare i64 @checked_unary_acoth_UnsignedInteger64(i64)

declare i16 @unary_acoth_Real16(i16)

declare float @atanhf(float)

declare i16 @checked_unary_acoth_Real16(i16)

declare float @unary_acoth_Real32(float)

declare float @checked_unary_acoth_Real32(float)

declare double @unary_acoth_Real64(double)

declare double @checked_unary_acoth_Real64(double)

declare void @unary_acoth_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_acoth_ComplexReal32(float*, float*, float, float)

declare void @unary_acoth_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_acoth_ComplexReal64(double*, double*, double, double)

declare i8 @unary_acsc_Integer8(i8)

declare i8 @checked_unary_acsc_Integer8(i8)

declare i16 @unary_acsc_Integer16(i16)

declare i16 @checked_unary_acsc_Integer16(i16)

declare i32 @unary_acsc_Integer32(i32)

declare i32 @checked_unary_acsc_Integer32(i32)

declare i64 @unary_acsc_Integer64(i64)

declare i64 @checked_unary_acsc_Integer64(i64)

declare i8 @unary_acsc_UnsignedInteger8(i8)

declare i8 @checked_unary_acsc_UnsignedInteger8(i8)

declare i16 @unary_acsc_UnsignedInteger16(i16)

declare i16 @checked_unary_acsc_UnsignedInteger16(i16)

declare i32 @unary_acsc_UnsignedInteger32(i32)

declare i32 @checked_unary_acsc_UnsignedInteger32(i32)

declare i64 @unary_acsc_UnsignedInteger64(i64)

declare i64 @checked_unary_acsc_UnsignedInteger64(i64)

declare i16 @unary_acsc_Real16(i16)

declare float @asinf(float)

declare i16 @checked_unary_acsc_Real16(i16)

declare float @unary_acsc_Real32(float)

declare float @checked_unary_acsc_Real32(float)

declare double @unary_acsc_Real64(double)

declare double @checked_unary_acsc_Real64(double)

declare void @unary_acsc_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_acsc_ComplexReal32(float*, float*, float, float)

declare void @unary_acsc_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_acsc_ComplexReal64(double*, double*, double, double)

declare i8 @unary_acsch_Integer8(i8)

declare i8 @checked_unary_acsch_Integer8(i8)

declare i16 @unary_acsch_Integer16(i16)

declare i16 @checked_unary_acsch_Integer16(i16)

declare i32 @unary_acsch_Integer32(i32)

declare i32 @checked_unary_acsch_Integer32(i32)

declare i64 @unary_acsch_Integer64(i64)

declare i64 @checked_unary_acsch_Integer64(i64)

declare i8 @unary_acsch_UnsignedInteger8(i8)

declare i8 @checked_unary_acsch_UnsignedInteger8(i8)

declare i16 @unary_acsch_UnsignedInteger16(i16)

declare i16 @checked_unary_acsch_UnsignedInteger16(i16)

declare i32 @unary_acsch_UnsignedInteger32(i32)

declare i32 @checked_unary_acsch_UnsignedInteger32(i32)

declare i64 @unary_acsch_UnsignedInteger64(i64)

declare i64 @checked_unary_acsch_UnsignedInteger64(i64)

declare i16 @unary_acsch_Real16(i16)

declare float @asinhf(float)

declare i16 @checked_unary_acsch_Real16(i16)

declare float @unary_acsch_Real32(float)

declare float @checked_unary_acsch_Real32(float)

declare double @unary_acsch_Real64(double)

declare double @checked_unary_acsch_Real64(double)

declare void @unary_acsch_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_acsch_ComplexReal32(float*, float*, float, float)

declare void @unary_acsch_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_acsch_ComplexReal64(double*, double*, double, double)

declare i8 @unary_arg_Integer8(i8)

declare i8 @checked_unary_arg_Integer8(i8)

declare i16 @unary_arg_Integer16(i16)

declare i16 @checked_unary_arg_Integer16(i16)

declare i32 @unary_arg_Integer32(i32)

declare i32 @checked_unary_arg_Integer32(i32)

declare i64 @unary_arg_Integer64(i64)

declare i64 @checked_unary_arg_Integer64(i64)

declare i8 @unary_arg_UnsignedInteger8(i8)

declare i8 @checked_unary_arg_UnsignedInteger8(i8)

declare i16 @unary_arg_UnsignedInteger16(i16)

declare i16 @checked_unary_arg_UnsignedInteger16(i16)

declare i32 @unary_arg_UnsignedInteger32(i32)

declare i32 @checked_unary_arg_UnsignedInteger32(i32)

declare i64 @unary_arg_UnsignedInteger64(i64)

declare i64 @checked_unary_arg_UnsignedInteger64(i64)

declare i64 @unary_arg_Real16(i16)

declare i64 @checked_unary_arg_Real16(i16)

declare i64 @unary_arg_Real32(float)

declare i64 @checked_unary_arg_Real32(float)

declare i64 @unary_arg_Real64(double)

declare i64 @checked_unary_arg_Real64(double)

declare float @unary_arg_ComplexReal32(float, float)

declare float @checked_unary_arg_ComplexReal32(float, float)

declare double @unary_arg_ComplexReal64(double, double)

declare double @checked_unary_arg_ComplexReal64(double, double)

declare i8 @unary_asec_Integer8(i8)

declare i8 @checked_unary_asec_Integer8(i8)

declare i16 @unary_asec_Integer16(i16)

declare i16 @checked_unary_asec_Integer16(i16)

declare i32 @unary_asec_Integer32(i32)

declare i32 @checked_unary_asec_Integer32(i32)

declare i64 @unary_asec_Integer64(i64)

declare i64 @checked_unary_asec_Integer64(i64)

declare i8 @unary_asec_UnsignedInteger8(i8)

declare i8 @checked_unary_asec_UnsignedInteger8(i8)

declare i16 @unary_asec_UnsignedInteger16(i16)

declare i16 @checked_unary_asec_UnsignedInteger16(i16)

declare i32 @unary_asec_UnsignedInteger32(i32)

declare i32 @checked_unary_asec_UnsignedInteger32(i32)

declare i64 @unary_asec_UnsignedInteger64(i64)

declare i64 @checked_unary_asec_UnsignedInteger64(i64)

declare i16 @unary_asec_Real16(i16)

declare i16 @checked_unary_asec_Real16(i16)

declare float @unary_asec_Real32(float)

declare float @checked_unary_asec_Real32(float)

declare double @unary_asec_Real64(double)

declare double @checked_unary_asec_Real64(double)

declare void @unary_asec_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_asec_ComplexReal32(float*, float*, float, float)

declare void @unary_asec_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_asec_ComplexReal64(double*, double*, double, double)

declare i8 @unary_asech_Integer8(i8)

declare i8 @checked_unary_asech_Integer8(i8)

declare i16 @unary_asech_Integer16(i16)

declare i16 @checked_unary_asech_Integer16(i16)

declare i32 @unary_asech_Integer32(i32)

declare i32 @checked_unary_asech_Integer32(i32)

declare i64 @unary_asech_Integer64(i64)

declare i64 @checked_unary_asech_Integer64(i64)

declare i8 @unary_asech_UnsignedInteger8(i8)

declare i8 @checked_unary_asech_UnsignedInteger8(i8)

declare i16 @unary_asech_UnsignedInteger16(i16)

declare i16 @checked_unary_asech_UnsignedInteger16(i16)

declare i32 @unary_asech_UnsignedInteger32(i32)

declare i32 @checked_unary_asech_UnsignedInteger32(i32)

declare i64 @unary_asech_UnsignedInteger64(i64)

declare i64 @checked_unary_asech_UnsignedInteger64(i64)

declare i16 @unary_asech_Real16(i16)

declare i16 @checked_unary_asech_Real16(i16)

declare float @unary_asech_Real32(float)

declare float @checked_unary_asech_Real32(float)

declare double @unary_asech_Real64(double)

declare double @checked_unary_asech_Real64(double)

declare void @unary_asech_ComplexReal32(float*, float*, float, float)

declare <2 x float> @_ZN3wrt9scalar_op5unaryL8acosh_opINSt3__17complexIfEELNS_13runtime_flagsE0EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE4EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE(%"class.std::__1::complex.156"*)

declare void @checked_unary_asech_ComplexReal32(float*, float*, float, float)

declare void @unary_asech_ComplexReal64(double*, double*, double, double)

declare { double, double } @_ZN3wrt9scalar_op5unaryL8acosh_opINSt3__17complexIdEELNS_13runtime_flagsE0EEENS3_9enable_ifIXsr10is_complexIT_EE5valueENS1_6detail7op_infoILNS9_2opE4EN7rt_typeIS8_E4typeEE11result_typeEE4typeERKNSF_13argument_typeE.212(%"class.std::__1::complex"*)

declare void @checked_unary_asech_ComplexReal64(double*, double*, double, double)

declare i8 @unary_asin_Integer8(i8)

declare i8 @checked_unary_asin_Integer8(i8)

declare i16 @unary_asin_Integer16(i16)

declare i16 @checked_unary_asin_Integer16(i16)

declare i32 @unary_asin_Integer32(i32)

declare i32 @checked_unary_asin_Integer32(i32)

declare i64 @unary_asin_Integer64(i64)

declare i64 @checked_unary_asin_Integer64(i64)

declare i8 @unary_asin_UnsignedInteger8(i8)

declare i8 @checked_unary_asin_UnsignedInteger8(i8)

declare i16 @unary_asin_UnsignedInteger16(i16)

declare i16 @checked_unary_asin_UnsignedInteger16(i16)

declare i32 @unary_asin_UnsignedInteger32(i32)

declare i32 @checked_unary_asin_UnsignedInteger32(i32)

declare i64 @unary_asin_UnsignedInteger64(i64)

declare i64 @checked_unary_asin_UnsignedInteger64(i64)

declare i16 @unary_asin_Real16(i16)

declare i16 @checked_unary_asin_Real16(i16)

declare float @unary_asin_Real32(float)

declare float @checked_unary_asin_Real32(float)

declare double @unary_asin_Real64(double)

declare double @checked_unary_asin_Real64(double)

declare void @unary_asin_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_asin_ComplexReal32(float*, float*, float, float)

declare void @unary_asin_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_asin_ComplexReal64(double*, double*, double, double)

declare i8 @unary_asinh_Integer8(i8)

declare i8 @checked_unary_asinh_Integer8(i8)

declare i16 @unary_asinh_Integer16(i16)

declare i16 @checked_unary_asinh_Integer16(i16)

declare i32 @unary_asinh_Integer32(i32)

declare i32 @checked_unary_asinh_Integer32(i32)

declare i64 @unary_asinh_Integer64(i64)

declare i64 @checked_unary_asinh_Integer64(i64)

declare i8 @unary_asinh_UnsignedInteger8(i8)

declare i8 @checked_unary_asinh_UnsignedInteger8(i8)

declare i16 @unary_asinh_UnsignedInteger16(i16)

declare i16 @checked_unary_asinh_UnsignedInteger16(i16)

declare i32 @unary_asinh_UnsignedInteger32(i32)

declare i32 @checked_unary_asinh_UnsignedInteger32(i32)

declare i64 @unary_asinh_UnsignedInteger64(i64)

declare i64 @checked_unary_asinh_UnsignedInteger64(i64)

declare i16 @unary_asinh_Real16(i16)

declare i16 @checked_unary_asinh_Real16(i16)

declare float @unary_asinh_Real32(float)

declare float @checked_unary_asinh_Real32(float)

declare double @unary_asinh_Real64(double)

declare double @checked_unary_asinh_Real64(double)

declare void @unary_asinh_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_asinh_ComplexReal32(float*, float*, float, float)

declare void @unary_asinh_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_asinh_ComplexReal64(double*, double*, double, double)

declare i8 @unary_atan_Integer8(i8)

declare i8 @checked_unary_atan_Integer8(i8)

declare i16 @unary_atan_Integer16(i16)

declare i16 @checked_unary_atan_Integer16(i16)

declare i32 @unary_atan_Integer32(i32)

declare i32 @checked_unary_atan_Integer32(i32)

declare i64 @unary_atan_Integer64(i64)

declare i64 @checked_unary_atan_Integer64(i64)

declare i8 @unary_atan_UnsignedInteger8(i8)

declare i8 @checked_unary_atan_UnsignedInteger8(i8)

declare i16 @unary_atan_UnsignedInteger16(i16)

declare i16 @checked_unary_atan_UnsignedInteger16(i16)

declare i32 @unary_atan_UnsignedInteger32(i32)

declare i32 @checked_unary_atan_UnsignedInteger32(i32)

declare i64 @unary_atan_UnsignedInteger64(i64)

declare i64 @checked_unary_atan_UnsignedInteger64(i64)

declare i16 @unary_atan_Real16(i16)

declare i16 @checked_unary_atan_Real16(i16)

declare float @unary_atan_Real32(float)

declare float @checked_unary_atan_Real32(float)

declare double @unary_atan_Real64(double)

declare double @checked_unary_atan_Real64(double)

declare void @unary_atan_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_atan_ComplexReal32(float*, float*, float, float)

declare void @unary_atan_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_atan_ComplexReal64(double*, double*, double, double)

declare i8 @unary_atanh_Integer8(i8)

declare i8 @checked_unary_atanh_Integer8(i8)

declare i16 @unary_atanh_Integer16(i16)

declare i16 @checked_unary_atanh_Integer16(i16)

declare i32 @unary_atanh_Integer32(i32)

declare i32 @checked_unary_atanh_Integer32(i32)

declare i64 @unary_atanh_Integer64(i64)

declare i64 @checked_unary_atanh_Integer64(i64)

declare i8 @unary_atanh_UnsignedInteger8(i8)

declare i8 @checked_unary_atanh_UnsignedInteger8(i8)

declare i16 @unary_atanh_UnsignedInteger16(i16)

declare i16 @checked_unary_atanh_UnsignedInteger16(i16)

declare i32 @unary_atanh_UnsignedInteger32(i32)

declare i32 @checked_unary_atanh_UnsignedInteger32(i32)

declare i64 @unary_atanh_UnsignedInteger64(i64)

declare i64 @checked_unary_atanh_UnsignedInteger64(i64)

declare i16 @unary_atanh_Real16(i16)

declare i16 @checked_unary_atanh_Real16(i16)

declare float @unary_atanh_Real32(float)

declare float @checked_unary_atanh_Real32(float)

declare double @unary_atanh_Real64(double)

declare double @checked_unary_atanh_Real64(double)

declare void @unary_atanh_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_atanh_ComplexReal32(float*, float*, float, float)

declare void @unary_atanh_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_atanh_ComplexReal64(double*, double*, double, double)

declare i8 @unary_bit_count_Integer8(i8)

declare i8 @checked_unary_bit_count_Integer8(i8)

declare i16 @unary_bit_count_Integer16(i16)

declare i16 @checked_unary_bit_count_Integer16(i16)

declare i32 @unary_bit_count_Integer32(i32)

; Function Attrs: nounwind readnone speculatable
declare i32 @llvm.ctpop.i32(i32) #0

declare i32 @checked_unary_bit_count_Integer32(i32)

declare i64 @unary_bit_count_Integer64(i64)

declare i64 @checked_unary_bit_count_Integer64(i64)

declare i8 @unary_bit_count_UnsignedInteger8(i8)

declare i8 @checked_unary_bit_count_UnsignedInteger8(i8)

declare i16 @unary_bit_count_UnsignedInteger16(i16)

declare i16 @checked_unary_bit_count_UnsignedInteger16(i16)

declare i32 @unary_bit_count_UnsignedInteger32(i32)

declare i32 @checked_unary_bit_count_UnsignedInteger32(i32)

declare i64 @unary_bit_count_UnsignedInteger64(i64)

declare i64 @checked_unary_bit_count_UnsignedInteger64(i64)

declare i16 @unary_bit_count_Real16(i16)

declare i16 @checked_unary_bit_count_Real16(i16)

declare float @unary_bit_count_Real32(float)

declare float @checked_unary_bit_count_Real32(float)

declare double @unary_bit_count_Real64(double)

declare double @checked_unary_bit_count_Real64(double)

declare <2 x float> @unary_bit_count_ComplexReal32(<2 x float>)

declare <2 x float> @checked_unary_bit_count_ComplexReal32(<2 x float>)

declare { double, double } @unary_bit_count_ComplexReal64(double, double)

declare { double, double } @checked_unary_bit_count_ComplexReal64(double, double)

declare i8 @unary_bit_length_Integer8(i8)

declare i8 @checked_unary_bit_length_Integer8(i8)

declare i16 @unary_bit_length_Integer16(i16)

declare i16 @checked_unary_bit_length_Integer16(i16)

declare i32 @unary_bit_length_Integer32(i32)

declare i32 @checked_unary_bit_length_Integer32(i32)

declare i64 @unary_bit_length_Integer64(i64)

declare i64 @checked_unary_bit_length_Integer64(i64)

declare i8 @unary_bit_length_UnsignedInteger8(i8)

declare i8 @checked_unary_bit_length_UnsignedInteger8(i8)

declare i16 @unary_bit_length_UnsignedInteger16(i16)

declare i16 @checked_unary_bit_length_UnsignedInteger16(i16)

declare i32 @unary_bit_length_UnsignedInteger32(i32)

declare i32 @checked_unary_bit_length_UnsignedInteger32(i32)

declare i64 @unary_bit_length_UnsignedInteger64(i64)

declare i64 @checked_unary_bit_length_UnsignedInteger64(i64)

declare i16 @unary_bit_length_Real16(i16)

declare i16 @checked_unary_bit_length_Real16(i16)

declare float @unary_bit_length_Real32(float)

declare float @checked_unary_bit_length_Real32(float)

declare double @unary_bit_length_Real64(double)

declare double @checked_unary_bit_length_Real64(double)

declare <2 x float> @unary_bit_length_ComplexReal32(<2 x float>)

declare <2 x float> @checked_unary_bit_length_ComplexReal32(<2 x float>)

declare { double, double } @unary_bit_length_ComplexReal64(double, double)

declare { double, double } @checked_unary_bit_length_ComplexReal64(double, double)

declare i8 @unary_bit_not_Integer8(i8)

declare i8 @checked_unary_bit_not_Integer8(i8)

declare i16 @unary_bit_not_Integer16(i16)

declare i16 @checked_unary_bit_not_Integer16(i16)

declare i32 @unary_bit_not_Integer32(i32)

declare i32 @checked_unary_bit_not_Integer32(i32)

declare i64 @unary_bit_not_Integer64(i64)

declare i64 @checked_unary_bit_not_Integer64(i64)

declare i8 @unary_bit_not_UnsignedInteger8(i8)

declare i8 @checked_unary_bit_not_UnsignedInteger8(i8)

declare i16 @unary_bit_not_UnsignedInteger16(i16)

declare i16 @checked_unary_bit_not_UnsignedInteger16(i16)

declare i32 @unary_bit_not_UnsignedInteger32(i32)

declare i32 @checked_unary_bit_not_UnsignedInteger32(i32)

declare i64 @unary_bit_not_UnsignedInteger64(i64)

declare i64 @checked_unary_bit_not_UnsignedInteger64(i64)

declare i16 @unary_bit_not_Real16(i16)

declare i16 @checked_unary_bit_not_Real16(i16)

declare float @unary_bit_not_Real32(float)

declare float @checked_unary_bit_not_Real32(float)

declare double @unary_bit_not_Real64(double)

declare double @checked_unary_bit_not_Real64(double)

declare <2 x float> @unary_bit_not_ComplexReal32(<2 x float>)

declare <2 x float> @checked_unary_bit_not_ComplexReal32(<2 x float>)

declare { double, double } @unary_bit_not_ComplexReal64(double, double)

declare { double, double } @checked_unary_bit_not_ComplexReal64(double, double)

declare i8 @unary_cbrt_Integer8(i8)

declare i8 @checked_unary_cbrt_Integer8(i8)

declare i16 @unary_cbrt_Integer16(i16)

declare i16 @checked_unary_cbrt_Integer16(i16)

declare i32 @unary_cbrt_Integer32(i32)

declare i32 @checked_unary_cbrt_Integer32(i32)

declare i64 @unary_cbrt_Integer64(i64)

declare i64 @checked_unary_cbrt_Integer64(i64)

declare i8 @unary_cbrt_UnsignedInteger8(i8)

declare i8 @checked_unary_cbrt_UnsignedInteger8(i8)

declare i16 @unary_cbrt_UnsignedInteger16(i16)

declare i16 @checked_unary_cbrt_UnsignedInteger16(i16)

declare i32 @unary_cbrt_UnsignedInteger32(i32)

declare i32 @checked_unary_cbrt_UnsignedInteger32(i32)

declare i64 @unary_cbrt_UnsignedInteger64(i64)

declare i64 @checked_unary_cbrt_UnsignedInteger64(i64)

declare i16 @unary_cbrt_Real16(i16)

declare float @cbrtf(float)

declare i16 @checked_unary_cbrt_Real16(i16)

declare float @unary_cbrt_Real32(float)

declare float @checked_unary_cbrt_Real32(float)

declare double @unary_cbrt_Real64(double)

declare double @checked_unary_cbrt_Real64(double)

declare <2 x float> @unary_cbrt_ComplexReal32(<2 x float>)

declare <2 x float> @checked_unary_cbrt_ComplexReal32(<2 x float>)

declare { double, double } @unary_cbrt_ComplexReal64(double, double)

declare { double, double } @checked_unary_cbrt_ComplexReal64(double, double)

declare i8 @unary_ceiling_Integer8(i8)

declare i8 @checked_unary_ceiling_Integer8(i8)

declare i16 @unary_ceiling_Integer16(i16)

declare i16 @checked_unary_ceiling_Integer16(i16)

declare i32 @unary_ceiling_Integer32(i32)

declare i32 @checked_unary_ceiling_Integer32(i32)

declare i64 @unary_ceiling_Integer64(i64)

declare i64 @checked_unary_ceiling_Integer64(i64)

declare i8 @unary_ceiling_UnsignedInteger8(i8)

declare i8 @checked_unary_ceiling_UnsignedInteger8(i8)

declare i16 @unary_ceiling_UnsignedInteger16(i16)

declare i16 @checked_unary_ceiling_UnsignedInteger16(i16)

declare i32 @unary_ceiling_UnsignedInteger32(i32)

declare i32 @checked_unary_ceiling_UnsignedInteger32(i32)

declare i64 @unary_ceiling_UnsignedInteger64(i64)

declare i64 @checked_unary_ceiling_UnsignedInteger64(i64)

declare i64 @unary_ceiling_Real16(i16)

declare i64 @checked_unary_ceiling_Real16(i16)

declare i64 @unary_ceiling_Real32(float)

; Function Attrs: nounwind readnone speculatable
declare float @llvm.ceil.f32(float) #0

declare i64 @checked_unary_ceiling_Real32(float)

declare i64 @unary_ceiling_Real64(double)

declare i64 @checked_unary_ceiling_Real64(double)

declare i64 @unary_ceiling_ComplexReal32(<2 x float>)

declare i64 @checked_unary_ceiling_ComplexReal32(<2 x float>)

declare i64 @unary_ceiling_ComplexReal64(double, double)

declare i64 @checked_unary_ceiling_ComplexReal64(double, double)

declare i8 @unary_conj_Integer8(i8)

declare i8 @checked_unary_conj_Integer8(i8)

declare i16 @unary_conj_Integer16(i16)

declare i16 @checked_unary_conj_Integer16(i16)

declare i32 @unary_conj_Integer32(i32)

declare i32 @checked_unary_conj_Integer32(i32)

declare i64 @unary_conj_Integer64(i64)

declare i64 @checked_unary_conj_Integer64(i64)

declare i8 @unary_conj_UnsignedInteger8(i8)

declare i8 @checked_unary_conj_UnsignedInteger8(i8)

declare i16 @unary_conj_UnsignedInteger16(i16)

declare i16 @checked_unary_conj_UnsignedInteger16(i16)

declare i32 @unary_conj_UnsignedInteger32(i32)

declare i32 @checked_unary_conj_UnsignedInteger32(i32)

declare i64 @unary_conj_UnsignedInteger64(i64)

declare i64 @checked_unary_conj_UnsignedInteger64(i64)

declare i16 @unary_conj_Real16(i16)

declare i16 @checked_unary_conj_Real16(i16)

declare float @unary_conj_Real32(float)

declare float @checked_unary_conj_Real32(float)

declare double @unary_conj_Real64(double)

declare double @checked_unary_conj_Real64(double)

declare void @unary_conj_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_conj_ComplexReal32(float*, float*, float, float)

declare void @unary_conj_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_conj_ComplexReal64(double*, double*, double, double)

declare i8 @unary_copy_Integer8(i8)

declare i8 @checked_unary_copy_Integer8(i8)

declare i16 @unary_copy_Integer16(i16)

declare i16 @checked_unary_copy_Integer16(i16)

declare i32 @unary_copy_Integer32(i32)

declare i32 @checked_unary_copy_Integer32(i32)

declare i64 @unary_copy_Integer64(i64)

declare i64 @checked_unary_copy_Integer64(i64)

declare i8 @unary_copy_UnsignedInteger8(i8)

declare i8 @checked_unary_copy_UnsignedInteger8(i8)

declare i16 @unary_copy_UnsignedInteger16(i16)

declare i16 @checked_unary_copy_UnsignedInteger16(i16)

declare i32 @unary_copy_UnsignedInteger32(i32)

declare i32 @checked_unary_copy_UnsignedInteger32(i32)

declare i64 @unary_copy_UnsignedInteger64(i64)

declare i64 @checked_unary_copy_UnsignedInteger64(i64)

declare i16 @unary_copy_Real16(i16)

declare i16 @checked_unary_copy_Real16(i16)

declare float @unary_copy_Real32(float)

declare float @checked_unary_copy_Real32(float)

declare double @unary_copy_Real64(double)

declare double @checked_unary_copy_Real64(double)

declare void @unary_copy_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_copy_ComplexReal32(float*, float*, float, float)

declare void @unary_copy_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_copy_ComplexReal64(double*, double*, double, double)

declare i8 @unary_cos_Integer8(i8)

declare i8 @checked_unary_cos_Integer8(i8)

declare i16 @unary_cos_Integer16(i16)

declare i16 @checked_unary_cos_Integer16(i16)

declare i32 @unary_cos_Integer32(i32)

declare i32 @checked_unary_cos_Integer32(i32)

declare i64 @unary_cos_Integer64(i64)

declare i64 @checked_unary_cos_Integer64(i64)

declare i8 @unary_cos_UnsignedInteger8(i8)

declare i8 @checked_unary_cos_UnsignedInteger8(i8)

declare i16 @unary_cos_UnsignedInteger16(i16)

declare i16 @checked_unary_cos_UnsignedInteger16(i16)

declare i32 @unary_cos_UnsignedInteger32(i32)

declare i32 @checked_unary_cos_UnsignedInteger32(i32)

declare i64 @unary_cos_UnsignedInteger64(i64)

declare i64 @checked_unary_cos_UnsignedInteger64(i64)

declare i16 @unary_cos_Real16(i16)

declare i16 @checked_unary_cos_Real16(i16)

declare float @unary_cos_Real32(float)

declare float @checked_unary_cos_Real32(float)

declare double @unary_cos_Real64(double)

declare double @checked_unary_cos_Real64(double)

declare void @unary_cos_ComplexReal32(float*, float*, float, float)

declare float @coshf(float)

declare float @sinhf(float)

declare void @checked_unary_cos_ComplexReal32(float*, float*, float, float)

declare void @unary_cos_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_cos_ComplexReal64(double*, double*, double, double)

declare i8 @unary_cosh_Integer8(i8)

declare i8 @checked_unary_cosh_Integer8(i8)

declare i16 @unary_cosh_Integer16(i16)

declare i16 @checked_unary_cosh_Integer16(i16)

declare i32 @unary_cosh_Integer32(i32)

declare i32 @checked_unary_cosh_Integer32(i32)

declare i64 @unary_cosh_Integer64(i64)

declare i64 @checked_unary_cosh_Integer64(i64)

declare i8 @unary_cosh_UnsignedInteger8(i8)

declare i8 @checked_unary_cosh_UnsignedInteger8(i8)

declare i16 @unary_cosh_UnsignedInteger16(i16)

declare i16 @checked_unary_cosh_UnsignedInteger16(i16)

declare i32 @unary_cosh_UnsignedInteger32(i32)

declare i32 @checked_unary_cosh_UnsignedInteger32(i32)

declare i64 @unary_cosh_UnsignedInteger64(i64)

declare i64 @checked_unary_cosh_UnsignedInteger64(i64)

declare i16 @unary_cosh_Real16(i16)

declare i16 @checked_unary_cosh_Real16(i16)

declare float @unary_cosh_Real32(float)

declare float @checked_unary_cosh_Real32(float)

declare double @unary_cosh_Real64(double)

declare double @checked_unary_cosh_Real64(double)

declare void @unary_cosh_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_cosh_ComplexReal32(float*, float*, float, float)

declare void @unary_cosh_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_cosh_ComplexReal64(double*, double*, double, double)

declare i8 @unary_cot_Integer8(i8)

declare i8 @checked_unary_cot_Integer8(i8)

declare i16 @unary_cot_Integer16(i16)

declare i16 @checked_unary_cot_Integer16(i16)

declare i32 @unary_cot_Integer32(i32)

declare i32 @checked_unary_cot_Integer32(i32)

declare i64 @unary_cot_Integer64(i64)

declare i64 @checked_unary_cot_Integer64(i64)

declare i8 @unary_cot_UnsignedInteger8(i8)

declare i8 @checked_unary_cot_UnsignedInteger8(i8)

declare i16 @unary_cot_UnsignedInteger16(i16)

declare i16 @checked_unary_cot_UnsignedInteger16(i16)

declare i32 @unary_cot_UnsignedInteger32(i32)

declare i32 @checked_unary_cot_UnsignedInteger32(i32)

declare i64 @unary_cot_UnsignedInteger64(i64)

declare i64 @checked_unary_cot_UnsignedInteger64(i64)

declare i16 @unary_cot_Real16(i16)

declare float @tanf(float)

declare i16 @checked_unary_cot_Real16(i16)

declare float @unary_cot_Real32(float)

declare float @checked_unary_cot_Real32(float)

declare double @unary_cot_Real64(double)

declare double @checked_unary_cot_Real64(double)

declare void @unary_cot_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_cot_ComplexReal32(float*, float*, float, float)

declare void @unary_cot_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_cot_ComplexReal64(double*, double*, double, double)

declare i8 @unary_coth_Integer8(i8)

declare i8 @checked_unary_coth_Integer8(i8)

declare i16 @unary_coth_Integer16(i16)

declare i16 @checked_unary_coth_Integer16(i16)

declare i32 @unary_coth_Integer32(i32)

declare i32 @checked_unary_coth_Integer32(i32)

declare i64 @unary_coth_Integer64(i64)

declare i64 @checked_unary_coth_Integer64(i64)

declare i8 @unary_coth_UnsignedInteger8(i8)

declare i8 @checked_unary_coth_UnsignedInteger8(i8)

declare i16 @unary_coth_UnsignedInteger16(i16)

declare i16 @checked_unary_coth_UnsignedInteger16(i16)

declare i32 @unary_coth_UnsignedInteger32(i32)

declare i32 @checked_unary_coth_UnsignedInteger32(i32)

declare i64 @unary_coth_UnsignedInteger64(i64)

declare i64 @checked_unary_coth_UnsignedInteger64(i64)

declare i16 @unary_coth_Real16(i16)

declare float @tanhf(float)

declare i16 @checked_unary_coth_Real16(i16)

declare float @unary_coth_Real32(float)

declare float @checked_unary_coth_Real32(float)

declare double @unary_coth_Real64(double)

declare double @checked_unary_coth_Real64(double)

declare void @unary_coth_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_coth_ComplexReal32(float*, float*, float, float)

declare void @unary_coth_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_coth_ComplexReal64(double*, double*, double, double)

declare i8 @unary_csc_Integer8(i8)

declare i8 @checked_unary_csc_Integer8(i8)

declare i16 @unary_csc_Integer16(i16)

declare i16 @checked_unary_csc_Integer16(i16)

declare i32 @unary_csc_Integer32(i32)

declare i32 @checked_unary_csc_Integer32(i32)

declare i64 @unary_csc_Integer64(i64)

declare i64 @checked_unary_csc_Integer64(i64)

declare i8 @unary_csc_UnsignedInteger8(i8)

declare i8 @checked_unary_csc_UnsignedInteger8(i8)

declare i16 @unary_csc_UnsignedInteger16(i16)

declare i16 @checked_unary_csc_UnsignedInteger16(i16)

declare i32 @unary_csc_UnsignedInteger32(i32)

declare i32 @checked_unary_csc_UnsignedInteger32(i32)

declare i64 @unary_csc_UnsignedInteger64(i64)

declare i64 @checked_unary_csc_UnsignedInteger64(i64)

declare i16 @unary_csc_Real16(i16)

declare i16 @checked_unary_csc_Real16(i16)

declare float @unary_csc_Real32(float)

declare float @checked_unary_csc_Real32(float)

declare double @unary_csc_Real64(double)

declare double @checked_unary_csc_Real64(double)

declare void @unary_csc_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_csc_ComplexReal32(float*, float*, float, float)

declare void @unary_csc_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_csc_ComplexReal64(double*, double*, double, double)

declare i8 @unary_csch_Integer8(i8)

declare i8 @checked_unary_csch_Integer8(i8)

declare i16 @unary_csch_Integer16(i16)

declare i16 @checked_unary_csch_Integer16(i16)

declare i32 @unary_csch_Integer32(i32)

declare i32 @checked_unary_csch_Integer32(i32)

declare i64 @unary_csch_Integer64(i64)

declare i64 @checked_unary_csch_Integer64(i64)

declare i8 @unary_csch_UnsignedInteger8(i8)

declare i8 @checked_unary_csch_UnsignedInteger8(i8)

declare i16 @unary_csch_UnsignedInteger16(i16)

declare i16 @checked_unary_csch_UnsignedInteger16(i16)

declare i32 @unary_csch_UnsignedInteger32(i32)

declare i32 @checked_unary_csch_UnsignedInteger32(i32)

declare i64 @unary_csch_UnsignedInteger64(i64)

declare i64 @checked_unary_csch_UnsignedInteger64(i64)

declare i16 @unary_csch_Real16(i16)

declare i16 @checked_unary_csch_Real16(i16)

declare float @unary_csch_Real32(float)

declare float @checked_unary_csch_Real32(float)

declare double @unary_csch_Real64(double)

declare double @checked_unary_csch_Real64(double)

declare void @unary_csch_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_csch_ComplexReal32(float*, float*, float, float)

declare void @unary_csch_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_csch_ComplexReal64(double*, double*, double, double)

declare i8 @unary_erf_Integer8(i8)

declare i8 @checked_unary_erf_Integer8(i8)

declare i16 @unary_erf_Integer16(i16)

declare i16 @checked_unary_erf_Integer16(i16)

declare i32 @unary_erf_Integer32(i32)

declare i32 @checked_unary_erf_Integer32(i32)

declare i64 @unary_erf_Integer64(i64)

declare i64 @checked_unary_erf_Integer64(i64)

declare i8 @unary_erf_UnsignedInteger8(i8)

declare i8 @checked_unary_erf_UnsignedInteger8(i8)

declare i16 @unary_erf_UnsignedInteger16(i16)

declare i16 @checked_unary_erf_UnsignedInteger16(i16)

declare i32 @unary_erf_UnsignedInteger32(i32)

declare i32 @checked_unary_erf_UnsignedInteger32(i32)

declare i64 @unary_erf_UnsignedInteger64(i64)

declare i64 @checked_unary_erf_UnsignedInteger64(i64)

declare i16 @unary_erf_Real16(i16)

declare float @erff(float)

declare i16 @checked_unary_erf_Real16(i16)

declare float @unary_erf_Real32(float)

declare float @checked_unary_erf_Real32(float)

declare double @unary_erf_Real64(double)

declare double @checked_unary_erf_Real64(double)

declare void @unary_erf_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_erf_ComplexReal32(float*, float*, float, float)

declare void @unary_erf_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_erf_ComplexReal64(double*, double*, double, double)

declare i8 @unary_erfc_Integer8(i8)

declare i8 @checked_unary_erfc_Integer8(i8)

declare i16 @unary_erfc_Integer16(i16)

declare i16 @checked_unary_erfc_Integer16(i16)

declare i32 @unary_erfc_Integer32(i32)

declare i32 @checked_unary_erfc_Integer32(i32)

declare i64 @unary_erfc_Integer64(i64)

declare i64 @checked_unary_erfc_Integer64(i64)

declare i8 @unary_erfc_UnsignedInteger8(i8)

declare i8 @checked_unary_erfc_UnsignedInteger8(i8)

declare i16 @unary_erfc_UnsignedInteger16(i16)

declare i16 @checked_unary_erfc_UnsignedInteger16(i16)

declare i32 @unary_erfc_UnsignedInteger32(i32)

declare i32 @checked_unary_erfc_UnsignedInteger32(i32)

declare i64 @unary_erfc_UnsignedInteger64(i64)

declare i64 @checked_unary_erfc_UnsignedInteger64(i64)

declare i16 @unary_erfc_Real16(i16)

declare float @erfcf(float)

declare i16 @checked_unary_erfc_Real16(i16)

declare float @unary_erfc_Real32(float)

declare float @checked_unary_erfc_Real32(float)

declare double @unary_erfc_Real64(double)

declare double @checked_unary_erfc_Real64(double)

declare void @unary_erfc_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_erfc_ComplexReal32(float*, float*, float, float)

declare void @unary_erfc_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_erfc_ComplexReal64(double*, double*, double, double)

declare i64 @unary_evenq_Integer8(i8)

declare i64 @checked_unary_evenq_Integer8(i8)

declare i64 @unary_evenq_Integer16(i16)

declare i64 @checked_unary_evenq_Integer16(i16)

declare i64 @unary_evenq_Integer32(i32)

declare i64 @checked_unary_evenq_Integer32(i32)

declare i64 @unary_evenq_Integer64(i64)

declare i64 @checked_unary_evenq_Integer64(i64)

declare i64 @unary_evenq_UnsignedInteger8(i8)

declare i64 @checked_unary_evenq_UnsignedInteger8(i8)

declare i64 @unary_evenq_UnsignedInteger16(i16)

declare i64 @checked_unary_evenq_UnsignedInteger16(i16)

declare i64 @unary_evenq_UnsignedInteger32(i32)

declare i64 @checked_unary_evenq_UnsignedInteger32(i32)

declare i64 @unary_evenq_UnsignedInteger64(i64)

declare i64 @checked_unary_evenq_UnsignedInteger64(i64)

declare i64 @unary_evenq_Real16(i16)

declare i64 @checked_unary_evenq_Real16(i16)

declare i64 @unary_evenq_Real32(float)

declare i64 @checked_unary_evenq_Real32(float)

declare i64 @unary_evenq_Real64(double)

declare i64 @checked_unary_evenq_Real64(double)

declare i64 @unary_evenq_ComplexReal32(<2 x float>)

declare i64 @checked_unary_evenq_ComplexReal32(<2 x float>)

declare i64 @unary_evenq_ComplexReal64(double, double)

declare i64 @checked_unary_evenq_ComplexReal64(double, double)

declare i8 @unary_exp_Integer8(i8)

declare i8 @checked_unary_exp_Integer8(i8)

declare i16 @unary_exp_Integer16(i16)

declare i16 @checked_unary_exp_Integer16(i16)

declare i32 @unary_exp_Integer32(i32)

declare i32 @checked_unary_exp_Integer32(i32)

declare i64 @unary_exp_Integer64(i64)

declare i64 @checked_unary_exp_Integer64(i64)

declare i8 @unary_exp_UnsignedInteger8(i8)

declare i8 @checked_unary_exp_UnsignedInteger8(i8)

declare i16 @unary_exp_UnsignedInteger16(i16)

declare i16 @checked_unary_exp_UnsignedInteger16(i16)

declare i32 @unary_exp_UnsignedInteger32(i32)

declare i32 @checked_unary_exp_UnsignedInteger32(i32)

declare i64 @unary_exp_UnsignedInteger64(i64)

declare i64 @checked_unary_exp_UnsignedInteger64(i64)

declare i16 @unary_exp_Real16(i16)

declare i16 @checked_unary_exp_Real16(i16)

declare float @unary_exp_Real32(float)

declare float @checked_unary_exp_Real32(float)

declare double @unary_exp_Real64(double)

declare double @checked_unary_exp_Real64(double)

declare void @unary_exp_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_exp_ComplexReal32(float*, float*, float, float)

declare void @unary_exp_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_exp_ComplexReal64(double*, double*, double, double)

declare i8 @unary_expm1_Integer8(i8)

declare i8 @checked_unary_expm1_Integer8(i8)

declare i16 @unary_expm1_Integer16(i16)

declare i16 @checked_unary_expm1_Integer16(i16)

declare i32 @unary_expm1_Integer32(i32)

declare i32 @checked_unary_expm1_Integer32(i32)

declare i64 @unary_expm1_Integer64(i64)

declare i64 @checked_unary_expm1_Integer64(i64)

declare i8 @unary_expm1_UnsignedInteger8(i8)

declare i8 @checked_unary_expm1_UnsignedInteger8(i8)

declare i16 @unary_expm1_UnsignedInteger16(i16)

declare i16 @checked_unary_expm1_UnsignedInteger16(i16)

declare i32 @unary_expm1_UnsignedInteger32(i32)

declare i32 @checked_unary_expm1_UnsignedInteger32(i32)

declare i64 @unary_expm1_UnsignedInteger64(i64)

declare i64 @checked_unary_expm1_UnsignedInteger64(i64)

declare i16 @unary_expm1_Real16(i16)

declare float @expm1f(float)

declare i16 @checked_unary_expm1_Real16(i16)

declare float @unary_expm1_Real32(float)

declare float @checked_unary_expm1_Real32(float)

declare double @unary_expm1_Real64(double)

declare double @checked_unary_expm1_Real64(double)

declare void @unary_expm1_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_expm1_ComplexReal32(float*, float*, float, float)

declare void @unary_expm1_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_expm1_ComplexReal64(double*, double*, double, double)

declare i8 @unary_fibonacci_Integer8(i8)

declare i8 @checked_unary_fibonacci_Integer8(i8)

declare i16 @unary_fibonacci_Integer16(i16)

declare i16 @checked_unary_fibonacci_Integer16(i16)

declare i32 @unary_fibonacci_Integer32(i32)

declare i32 @checked_unary_fibonacci_Integer32(i32)

declare i64 @unary_fibonacci_Integer64(i64)

declare i64 @checked_unary_fibonacci_Integer64(i64)

declare i8 @unary_fibonacci_UnsignedInteger8(i8)

declare i8 @checked_unary_fibonacci_UnsignedInteger8(i8)

declare i16 @unary_fibonacci_UnsignedInteger16(i16)

declare i16 @checked_unary_fibonacci_UnsignedInteger16(i16)

declare i32 @unary_fibonacci_UnsignedInteger32(i32)

declare i32 @checked_unary_fibonacci_UnsignedInteger32(i32)

declare i64 @unary_fibonacci_UnsignedInteger64(i64)

declare i64 @checked_unary_fibonacci_UnsignedInteger64(i64)

declare i16 @unary_fibonacci_Real16(i16)

declare i16 @checked_unary_fibonacci_Real16(i16)

declare float @unary_fibonacci_Real32(float)

declare float @checked_unary_fibonacci_Real32(float)

declare double @unary_fibonacci_Real64(double)

declare double @checked_unary_fibonacci_Real64(double)

declare void @unary_fibonacci_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_fibonacci_ComplexReal32(float*, float*, float, float)

declare void @unary_fibonacci_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_fibonacci_ComplexReal64(double*, double*, double, double)

declare i8 @unary_floor_Integer8(i8)

declare i8 @checked_unary_floor_Integer8(i8)

declare i16 @unary_floor_Integer16(i16)

declare i16 @checked_unary_floor_Integer16(i16)

declare i32 @unary_floor_Integer32(i32)

declare i32 @checked_unary_floor_Integer32(i32)

declare i64 @unary_floor_Integer64(i64)

declare i64 @checked_unary_floor_Integer64(i64)

declare i8 @unary_floor_UnsignedInteger8(i8)

declare i8 @checked_unary_floor_UnsignedInteger8(i8)

declare i16 @unary_floor_UnsignedInteger16(i16)

declare i16 @checked_unary_floor_UnsignedInteger16(i16)

declare i32 @unary_floor_UnsignedInteger32(i32)

declare i32 @checked_unary_floor_UnsignedInteger32(i32)

declare i64 @unary_floor_UnsignedInteger64(i64)

declare i64 @checked_unary_floor_UnsignedInteger64(i64)

declare i64 @unary_floor_Real16(i16)

declare i64 @checked_unary_floor_Real16(i16)

declare i64 @unary_floor_Real32(float)

declare i64 @checked_unary_floor_Real32(float)

declare i64 @unary_floor_Real64(double)

declare i64 @checked_unary_floor_Real64(double)

declare i64 @unary_floor_ComplexReal32(<2 x float>)

declare i64 @checked_unary_floor_ComplexReal32(<2 x float>)

declare i64 @unary_floor_ComplexReal64(double, double)

declare i64 @checked_unary_floor_ComplexReal64(double, double)

declare i64 @unary_fpexception_Integer8(i8)

declare i64 @checked_unary_fpexception_Integer8(i8)

declare i64 @unary_fpexception_Integer16(i16)

declare i64 @checked_unary_fpexception_Integer16(i16)

declare i64 @unary_fpexception_Integer32(i32)

declare i64 @checked_unary_fpexception_Integer32(i32)

declare i64 @unary_fpexception_Integer64(i64)

declare i64 @checked_unary_fpexception_Integer64(i64)

declare i64 @unary_fpexception_UnsignedInteger8(i8)

declare i64 @checked_unary_fpexception_UnsignedInteger8(i8)

declare i64 @unary_fpexception_UnsignedInteger16(i16)

declare i64 @checked_unary_fpexception_UnsignedInteger16(i16)

declare i64 @unary_fpexception_UnsignedInteger32(i32)

declare i64 @checked_unary_fpexception_UnsignedInteger32(i32)

declare i64 @unary_fpexception_UnsignedInteger64(i64)

declare i64 @checked_unary_fpexception_UnsignedInteger64(i64)

declare i64 @unary_fpexception_Real16(i16)

declare i64 @checked_unary_fpexception_Real16(i16)

declare i64 @unary_fpexception_Real32(float)

declare i64 @checked_unary_fpexception_Real32(float)

declare i64 @unary_fpexception_Real64(double)

declare i64 @checked_unary_fpexception_Real64(double)

declare i64 @unary_fpexception_ComplexReal32(<2 x float>)

declare i64 @checked_unary_fpexception_ComplexReal32(<2 x float>)

declare i64 @unary_fpexception_ComplexReal64(double, double)

declare i64 @checked_unary_fpexception_ComplexReal64(double, double)

declare i8 @unary_fracpart_Integer8(i8)

declare i8 @checked_unary_fracpart_Integer8(i8)

declare i16 @unary_fracpart_Integer16(i16)

declare i16 @checked_unary_fracpart_Integer16(i16)

declare i32 @unary_fracpart_Integer32(i32)

declare i32 @checked_unary_fracpart_Integer32(i32)

declare i64 @unary_fracpart_Integer64(i64)

declare i64 @checked_unary_fracpart_Integer64(i64)

declare i8 @unary_fracpart_UnsignedInteger8(i8)

declare i8 @checked_unary_fracpart_UnsignedInteger8(i8)

declare i16 @unary_fracpart_UnsignedInteger16(i16)

declare i16 @checked_unary_fracpart_UnsignedInteger16(i16)

declare i32 @unary_fracpart_UnsignedInteger32(i32)

declare i32 @checked_unary_fracpart_UnsignedInteger32(i32)

declare i64 @unary_fracpart_UnsignedInteger64(i64)

declare i64 @checked_unary_fracpart_UnsignedInteger64(i64)

declare i16 @unary_fracpart_Real16(i16)

declare i16 @checked_unary_fracpart_Real16(i16)

declare float @unary_fracpart_Real32(float)

declare float @checked_unary_fracpart_Real32(float)

declare double @unary_fracpart_Real64(double)

declare double @checked_unary_fracpart_Real64(double)

declare void @unary_fracpart_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_fracpart_ComplexReal32(float*, float*, float, float)

declare void @unary_fracpart_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_fracpart_ComplexReal64(double*, double*, double, double)

declare i8 @unary_gamma_Integer8(i8)

declare i8 @checked_unary_gamma_Integer8(i8)

declare i16 @unary_gamma_Integer16(i16)

declare i16 @checked_unary_gamma_Integer16(i16)

declare i32 @unary_gamma_Integer32(i32)

declare i32 @checked_unary_gamma_Integer32(i32)

declare i64 @unary_gamma_Integer64(i64)

declare i64 @checked_unary_gamma_Integer64(i64)

declare i8 @unary_gamma_UnsignedInteger8(i8)

declare i8 @checked_unary_gamma_UnsignedInteger8(i8)

declare i16 @unary_gamma_UnsignedInteger16(i16)

declare i16 @checked_unary_gamma_UnsignedInteger16(i16)

declare i32 @unary_gamma_UnsignedInteger32(i32)

declare i32 @checked_unary_gamma_UnsignedInteger32(i32)

declare i64 @unary_gamma_UnsignedInteger64(i64)

declare i64 @checked_unary_gamma_UnsignedInteger64(i64)

declare i16 @unary_gamma_Real16(i16)

declare float @tgammaf(float)

declare i16 @checked_unary_gamma_Real16(i16)

declare float @unary_gamma_Real32(float)

declare float @checked_unary_gamma_Real32(float)

declare double @unary_gamma_Real64(double)

declare double @checked_unary_gamma_Real64(double)

declare void @unary_gamma_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_gamma_ComplexReal32(float*, float*, float, float)

declare void @unary_gamma_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_gamma_ComplexReal64(double*, double*, double, double)

declare i8 @unary_gudermannian_Integer8(i8)

declare i8 @checked_unary_gudermannian_Integer8(i8)

declare i16 @unary_gudermannian_Integer16(i16)

declare i16 @checked_unary_gudermannian_Integer16(i16)

declare i32 @unary_gudermannian_Integer32(i32)

declare i32 @checked_unary_gudermannian_Integer32(i32)

declare i64 @unary_gudermannian_Integer64(i64)

declare i64 @checked_unary_gudermannian_Integer64(i64)

declare i8 @unary_gudermannian_UnsignedInteger8(i8)

declare i8 @checked_unary_gudermannian_UnsignedInteger8(i8)

declare i16 @unary_gudermannian_UnsignedInteger16(i16)

declare i16 @checked_unary_gudermannian_UnsignedInteger16(i16)

declare i32 @unary_gudermannian_UnsignedInteger32(i32)

declare i32 @checked_unary_gudermannian_UnsignedInteger32(i32)

declare i64 @unary_gudermannian_UnsignedInteger64(i64)

declare i64 @checked_unary_gudermannian_UnsignedInteger64(i64)

declare i16 @unary_gudermannian_Real16(i16)

declare i16 @checked_unary_gudermannian_Real16(i16)

declare float @unary_gudermannian_Real32(float)

declare float @checked_unary_gudermannian_Real32(float)

declare double @unary_gudermannian_Real64(double)

declare double @checked_unary_gudermannian_Real64(double)

declare void @unary_gudermannian_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_gudermannian_ComplexReal32(float*, float*, float, float)

declare void @unary_gudermannian_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_gudermannian_ComplexReal64(double*, double*, double, double)

declare i8 @unary_haversine_Integer8(i8)

declare i8 @checked_unary_haversine_Integer8(i8)

declare i16 @unary_haversine_Integer16(i16)

declare i16 @checked_unary_haversine_Integer16(i16)

declare i32 @unary_haversine_Integer32(i32)

declare i32 @checked_unary_haversine_Integer32(i32)

declare i64 @unary_haversine_Integer64(i64)

declare i64 @checked_unary_haversine_Integer64(i64)

declare i8 @unary_haversine_UnsignedInteger8(i8)

declare i8 @checked_unary_haversine_UnsignedInteger8(i8)

declare i16 @unary_haversine_UnsignedInteger16(i16)

declare i16 @checked_unary_haversine_UnsignedInteger16(i16)

declare i32 @unary_haversine_UnsignedInteger32(i32)

declare i32 @checked_unary_haversine_UnsignedInteger32(i32)

declare i64 @unary_haversine_UnsignedInteger64(i64)

declare i64 @checked_unary_haversine_UnsignedInteger64(i64)

declare i16 @unary_haversine_Real16(i16)

declare i16 @checked_unary_haversine_Real16(i16)

declare float @unary_haversine_Real32(float)

declare float @checked_unary_haversine_Real32(float)

declare double @unary_haversine_Real64(double)

declare double @checked_unary_haversine_Real64(double)

declare void @unary_haversine_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_haversine_ComplexReal32(float*, float*, float, float)

declare void @unary_haversine_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_haversine_ComplexReal64(double*, double*, double, double)

declare i8 @unary_im_Integer8(i8)

declare i8 @checked_unary_im_Integer8(i8)

declare i16 @unary_im_Integer16(i16)

declare i16 @checked_unary_im_Integer16(i16)

declare i32 @unary_im_Integer32(i32)

declare i32 @checked_unary_im_Integer32(i32)

declare i64 @unary_im_Integer64(i64)

declare i64 @checked_unary_im_Integer64(i64)

declare i8 @unary_im_UnsignedInteger8(i8)

declare i8 @checked_unary_im_UnsignedInteger8(i8)

declare i16 @unary_im_UnsignedInteger16(i16)

declare i16 @checked_unary_im_UnsignedInteger16(i16)

declare i32 @unary_im_UnsignedInteger32(i32)

declare i32 @checked_unary_im_UnsignedInteger32(i32)

declare i64 @unary_im_UnsignedInteger64(i64)

declare i64 @checked_unary_im_UnsignedInteger64(i64)

declare i64 @unary_im_Real16(i16)

declare i64 @checked_unary_im_Real16(i16)

declare i64 @unary_im_Real32(float)

declare i64 @checked_unary_im_Real32(float)

declare i64 @unary_im_Real64(double)

declare i64 @checked_unary_im_Real64(double)

declare float @unary_im_ComplexReal32(<2 x float>)

declare float @checked_unary_im_ComplexReal32(<2 x float>)

declare double @unary_im_ComplexReal64(double, double)

declare double @checked_unary_im_ComplexReal64(double, double)

declare i8 @unary_intpart_Integer8(i8)

declare i8 @checked_unary_intpart_Integer8(i8)

declare i16 @unary_intpart_Integer16(i16)

declare i16 @checked_unary_intpart_Integer16(i16)

declare i32 @unary_intpart_Integer32(i32)

declare i32 @checked_unary_intpart_Integer32(i32)

declare i64 @unary_intpart_Integer64(i64)

declare i64 @checked_unary_intpart_Integer64(i64)

declare i8 @unary_intpart_UnsignedInteger8(i8)

declare i8 @checked_unary_intpart_UnsignedInteger8(i8)

declare i16 @unary_intpart_UnsignedInteger16(i16)

declare i16 @checked_unary_intpart_UnsignedInteger16(i16)

declare i32 @unary_intpart_UnsignedInteger32(i32)

declare i32 @checked_unary_intpart_UnsignedInteger32(i32)

declare i64 @unary_intpart_UnsignedInteger64(i64)

declare i64 @checked_unary_intpart_UnsignedInteger64(i64)

declare i64 @unary_intpart_Real16(i16)

declare i64 @checked_unary_intpart_Real16(i16)

declare i64 @unary_intpart_Real32(float)

; Function Attrs: nounwind readnone speculatable
declare float @llvm.trunc.f32(float) #0

declare i64 @checked_unary_intpart_Real32(float)

declare i64 @unary_intpart_Real64(double)

declare i64 @checked_unary_intpart_Real64(double)

declare i64 @unary_intpart_ComplexReal32(<2 x float>)

declare i64 @checked_unary_intpart_ComplexReal32(<2 x float>)

declare i64 @unary_intpart_ComplexReal64(double, double)

declare i64 @checked_unary_intpart_ComplexReal64(double, double)

declare i8 @unary_inversegudermannian_Integer8(i8)

declare i8 @checked_unary_inversegudermannian_Integer8(i8)

declare i16 @unary_inversegudermannian_Integer16(i16)

declare i16 @checked_unary_inversegudermannian_Integer16(i16)

declare i32 @unary_inversegudermannian_Integer32(i32)

declare i32 @checked_unary_inversegudermannian_Integer32(i32)

declare i64 @unary_inversegudermannian_Integer64(i64)

declare i64 @checked_unary_inversegudermannian_Integer64(i64)

declare i8 @unary_inversegudermannian_UnsignedInteger8(i8)

declare i8 @checked_unary_inversegudermannian_UnsignedInteger8(i8)

declare i16 @unary_inversegudermannian_UnsignedInteger16(i16)

declare i16 @checked_unary_inversegudermannian_UnsignedInteger16(i16)

declare i32 @unary_inversegudermannian_UnsignedInteger32(i32)

declare i32 @checked_unary_inversegudermannian_UnsignedInteger32(i32)

declare i64 @unary_inversegudermannian_UnsignedInteger64(i64)

declare i64 @checked_unary_inversegudermannian_UnsignedInteger64(i64)

declare i16 @unary_inversegudermannian_Real16(i16)

declare i16 @checked_unary_inversegudermannian_Real16(i16)

declare float @unary_inversegudermannian_Real32(float)

declare float @checked_unary_inversegudermannian_Real32(float)

declare double @unary_inversegudermannian_Real64(double)

declare double @checked_unary_inversegudermannian_Real64(double)

declare void @unary_inversegudermannian_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_inversegudermannian_ComplexReal32(float*, float*, float, float)

declare void @unary_inversegudermannian_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_inversegudermannian_ComplexReal64(double*, double*, double, double)

declare i8 @unary_inversehaversine_Integer8(i8)

declare i8 @checked_unary_inversehaversine_Integer8(i8)

declare i16 @unary_inversehaversine_Integer16(i16)

declare i16 @checked_unary_inversehaversine_Integer16(i16)

declare i32 @unary_inversehaversine_Integer32(i32)

declare i32 @checked_unary_inversehaversine_Integer32(i32)

declare i64 @unary_inversehaversine_Integer64(i64)

declare i64 @checked_unary_inversehaversine_Integer64(i64)

declare i8 @unary_inversehaversine_UnsignedInteger8(i8)

declare i8 @checked_unary_inversehaversine_UnsignedInteger8(i8)

declare i16 @unary_inversehaversine_UnsignedInteger16(i16)

declare i16 @checked_unary_inversehaversine_UnsignedInteger16(i16)

declare i32 @unary_inversehaversine_UnsignedInteger32(i32)

declare i32 @checked_unary_inversehaversine_UnsignedInteger32(i32)

declare i64 @unary_inversehaversine_UnsignedInteger64(i64)

declare i64 @checked_unary_inversehaversine_UnsignedInteger64(i64)

declare i16 @unary_inversehaversine_Real16(i16)

declare i16 @checked_unary_inversehaversine_Real16(i16)

declare float @unary_inversehaversine_Real32(float)

declare float @checked_unary_inversehaversine_Real32(float)

declare double @unary_inversehaversine_Real64(double)

declare double @checked_unary_inversehaversine_Real64(double)

declare void @unary_inversehaversine_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_inversehaversine_ComplexReal32(float*, float*, float, float)

declare void @unary_inversehaversine_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_inversehaversine_ComplexReal64(double*, double*, double, double)

declare i8 @unary_log_Integer8(i8)

declare i8 @checked_unary_log_Integer8(i8)

declare i16 @unary_log_Integer16(i16)

declare i16 @checked_unary_log_Integer16(i16)

declare i32 @unary_log_Integer32(i32)

declare i32 @checked_unary_log_Integer32(i32)

declare i64 @unary_log_Integer64(i64)

declare i64 @checked_unary_log_Integer64(i64)

declare i8 @unary_log_UnsignedInteger8(i8)

declare i8 @checked_unary_log_UnsignedInteger8(i8)

declare i16 @unary_log_UnsignedInteger16(i16)

declare i16 @checked_unary_log_UnsignedInteger16(i16)

declare i32 @unary_log_UnsignedInteger32(i32)

declare i32 @checked_unary_log_UnsignedInteger32(i32)

declare i64 @unary_log_UnsignedInteger64(i64)

declare i64 @checked_unary_log_UnsignedInteger64(i64)

declare i16 @unary_log_Real16(i16)

declare i16 @checked_unary_log_Real16(i16)

declare float @unary_log_Real32(float)

declare float @checked_unary_log_Real32(float)

declare double @unary_log_Real64(double)

declare double @checked_unary_log_Real64(double)

declare void @unary_log_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_log_ComplexReal32(float*, float*, float, float)

declare void @unary_log_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_log_ComplexReal64(double*, double*, double, double)

declare i8 @unary_log10_Integer8(i8)

declare i8 @checked_unary_log10_Integer8(i8)

declare i16 @unary_log10_Integer16(i16)

declare i16 @checked_unary_log10_Integer16(i16)

declare i32 @unary_log10_Integer32(i32)

declare i32 @checked_unary_log10_Integer32(i32)

declare i64 @unary_log10_Integer64(i64)

declare i64 @checked_unary_log10_Integer64(i64)

declare i8 @unary_log10_UnsignedInteger8(i8)

declare i8 @checked_unary_log10_UnsignedInteger8(i8)

declare i16 @unary_log10_UnsignedInteger16(i16)

declare i16 @checked_unary_log10_UnsignedInteger16(i16)

declare i32 @unary_log10_UnsignedInteger32(i32)

declare i32 @checked_unary_log10_UnsignedInteger32(i32)

declare i64 @unary_log10_UnsignedInteger64(i64)

declare i64 @checked_unary_log10_UnsignedInteger64(i64)

declare i16 @unary_log10_Real16(i16)

; Function Attrs: nounwind readnone speculatable
declare float @llvm.log10.f32(float) #0

declare i16 @checked_unary_log10_Real16(i16)

declare float @unary_log10_Real32(float)

declare float @checked_unary_log10_Real32(float)

declare double @unary_log10_Real64(double)

declare double @checked_unary_log10_Real64(double)

declare void @unary_log10_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_log10_ComplexReal32(float*, float*, float, float)

declare void @unary_log10_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_log10_ComplexReal64(double*, double*, double, double)

declare i8 @unary_log1p_Integer8(i8)

declare i8 @checked_unary_log1p_Integer8(i8)

declare i16 @unary_log1p_Integer16(i16)

declare i16 @checked_unary_log1p_Integer16(i16)

declare i32 @unary_log1p_Integer32(i32)

declare i32 @checked_unary_log1p_Integer32(i32)

declare i64 @unary_log1p_Integer64(i64)

declare i64 @checked_unary_log1p_Integer64(i64)

declare i8 @unary_log1p_UnsignedInteger8(i8)

declare i8 @checked_unary_log1p_UnsignedInteger8(i8)

declare i16 @unary_log1p_UnsignedInteger16(i16)

declare i16 @checked_unary_log1p_UnsignedInteger16(i16)

declare i32 @unary_log1p_UnsignedInteger32(i32)

declare i32 @checked_unary_log1p_UnsignedInteger32(i32)

declare i64 @unary_log1p_UnsignedInteger64(i64)

declare i64 @checked_unary_log1p_UnsignedInteger64(i64)

declare i16 @unary_log1p_Real16(i16)

declare i16 @checked_unary_log1p_Real16(i16)

declare float @unary_log1p_Real32(float)

declare float @checked_unary_log1p_Real32(float)

declare double @unary_log1p_Real64(double)

declare double @checked_unary_log1p_Real64(double)

declare void @unary_log1p_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_log1p_ComplexReal32(float*, float*, float, float)

declare void @unary_log1p_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_log1p_ComplexReal64(double*, double*, double, double)

declare i8 @unary_log2_Integer8(i8)

declare i8 @checked_unary_log2_Integer8(i8)

declare i16 @unary_log2_Integer16(i16)

declare i16 @checked_unary_log2_Integer16(i16)

declare i32 @unary_log2_Integer32(i32)

declare i32 @checked_unary_log2_Integer32(i32)

declare i64 @unary_log2_Integer64(i64)

declare i64 @checked_unary_log2_Integer64(i64)

declare i8 @unary_log2_UnsignedInteger8(i8)

declare i8 @checked_unary_log2_UnsignedInteger8(i8)

declare i16 @unary_log2_UnsignedInteger16(i16)

declare i16 @checked_unary_log2_UnsignedInteger16(i16)

declare i32 @unary_log2_UnsignedInteger32(i32)

declare i32 @checked_unary_log2_UnsignedInteger32(i32)

declare i64 @unary_log2_UnsignedInteger64(i64)

declare i64 @checked_unary_log2_UnsignedInteger64(i64)

declare i16 @unary_log2_Real16(i16)

; Function Attrs: nounwind readnone speculatable
declare float @llvm.log2.f32(float) #0

declare i16 @checked_unary_log2_Real16(i16)

declare float @unary_log2_Real32(float)

declare float @checked_unary_log2_Real32(float)

declare double @unary_log2_Real64(double)

declare double @checked_unary_log2_Real64(double)

declare void @unary_log2_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_log2_ComplexReal32(float*, float*, float, float)

declare void @unary_log2_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_log2_ComplexReal64(double*, double*, double, double)

declare i8 @unary_loggamma_Integer8(i8)

declare i8 @checked_unary_loggamma_Integer8(i8)

declare i16 @unary_loggamma_Integer16(i16)

declare i16 @checked_unary_loggamma_Integer16(i16)

declare i32 @unary_loggamma_Integer32(i32)

declare i32 @checked_unary_loggamma_Integer32(i32)

declare i64 @unary_loggamma_Integer64(i64)

declare i64 @checked_unary_loggamma_Integer64(i64)

declare i8 @unary_loggamma_UnsignedInteger8(i8)

declare i8 @checked_unary_loggamma_UnsignedInteger8(i8)

declare i16 @unary_loggamma_UnsignedInteger16(i16)

declare i16 @checked_unary_loggamma_UnsignedInteger16(i16)

declare i32 @unary_loggamma_UnsignedInteger32(i32)

declare i32 @checked_unary_loggamma_UnsignedInteger32(i32)

declare i64 @unary_loggamma_UnsignedInteger64(i64)

declare i64 @checked_unary_loggamma_UnsignedInteger64(i64)

declare i16 @unary_loggamma_Real16(i16)

declare float @lgammaf(float)

declare i16 @checked_unary_loggamma_Real16(i16)

declare float @unary_loggamma_Real32(float)

declare float @checked_unary_loggamma_Real32(float)

declare double @unary_loggamma_Real64(double)

declare double @checked_unary_loggamma_Real64(double)

declare void @unary_loggamma_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_loggamma_ComplexReal32(float*, float*, float, float)

declare void @unary_loggamma_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_loggamma_ComplexReal64(double*, double*, double, double)

declare i8 @unary_logistic_Integer8(i8)

declare i8 @checked_unary_logistic_Integer8(i8)

declare i16 @unary_logistic_Integer16(i16)

declare i16 @checked_unary_logistic_Integer16(i16)

declare i32 @unary_logistic_Integer32(i32)

declare i32 @checked_unary_logistic_Integer32(i32)

declare i64 @unary_logistic_Integer64(i64)

declare i64 @checked_unary_logistic_Integer64(i64)

declare i8 @unary_logistic_UnsignedInteger8(i8)

declare i8 @checked_unary_logistic_UnsignedInteger8(i8)

declare i16 @unary_logistic_UnsignedInteger16(i16)

declare i16 @checked_unary_logistic_UnsignedInteger16(i16)

declare i32 @unary_logistic_UnsignedInteger32(i32)

declare i32 @checked_unary_logistic_UnsignedInteger32(i32)

declare i64 @unary_logistic_UnsignedInteger64(i64)

declare i64 @checked_unary_logistic_UnsignedInteger64(i64)

declare i16 @unary_logistic_Real16(i16)

declare i16 @checked_unary_logistic_Real16(i16)

declare float @unary_logistic_Real32(float)

declare float @checked_unary_logistic_Real32(float)

declare double @unary_logistic_Real64(double)

declare double @checked_unary_logistic_Real64(double)

declare void @unary_logistic_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_logistic_ComplexReal32(float*, float*, float, float)

declare void @unary_logistic_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_logistic_ComplexReal64(double*, double*, double, double)

declare i8 @unary_lucasl_Integer8(i8)

declare i8 @checked_unary_lucasl_Integer8(i8)

declare i16 @unary_lucasl_Integer16(i16)

declare i16 @checked_unary_lucasl_Integer16(i16)

declare i32 @unary_lucasl_Integer32(i32)

declare i32 @checked_unary_lucasl_Integer32(i32)

declare i64 @unary_lucasl_Integer64(i64)

declare i64 @checked_unary_lucasl_Integer64(i64)

declare i8 @unary_lucasl_UnsignedInteger8(i8)

declare i8 @checked_unary_lucasl_UnsignedInteger8(i8)

declare i16 @unary_lucasl_UnsignedInteger16(i16)

declare i16 @checked_unary_lucasl_UnsignedInteger16(i16)

declare i32 @unary_lucasl_UnsignedInteger32(i32)

declare i32 @checked_unary_lucasl_UnsignedInteger32(i32)

declare i64 @unary_lucasl_UnsignedInteger64(i64)

declare i64 @checked_unary_lucasl_UnsignedInteger64(i64)

declare i16 @unary_lucasl_Real16(i16)

declare i16 @checked_unary_lucasl_Real16(i16)

declare float @unary_lucasl_Real32(float)

declare float @checked_unary_lucasl_Real32(float)

declare double @unary_lucasl_Real64(double)

declare double @checked_unary_lucasl_Real64(double)

declare void @unary_lucasl_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_lucasl_ComplexReal32(float*, float*, float, float)

declare void @unary_lucasl_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_lucasl_ComplexReal64(double*, double*, double, double)

declare i8 @unary_minus_Integer8(i8)

declare i8 @checked_unary_minus_Integer8(i8)

declare i16 @unary_minus_Integer16(i16)

declare i16 @checked_unary_minus_Integer16(i16)

declare i32 @unary_minus_Integer32(i32)

declare i32 @checked_unary_minus_Integer32(i32)

declare i64 @unary_minus_Integer64(i64)

declare i64 @checked_unary_minus_Integer64(i64)

declare i8 @unary_minus_UnsignedInteger8(i8)

declare i8 @checked_unary_minus_UnsignedInteger8(i8)

declare i16 @unary_minus_UnsignedInteger16(i16)

declare i16 @checked_unary_minus_UnsignedInteger16(i16)

declare i32 @unary_minus_UnsignedInteger32(i32)

declare i32 @checked_unary_minus_UnsignedInteger32(i32)

declare i64 @unary_minus_UnsignedInteger64(i64)

declare i64 @checked_unary_minus_UnsignedInteger64(i64)

declare i16 @unary_minus_Real16(i16)

declare i16 @checked_unary_minus_Real16(i16)

declare float @unary_minus_Real32(float)

declare float @checked_unary_minus_Real32(float)

declare double @unary_minus_Real64(double)

declare double @checked_unary_minus_Real64(double)

declare void @unary_minus_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_minus_ComplexReal32(float*, float*, float, float)

declare void @unary_minus_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_minus_ComplexReal64(double*, double*, double, double)

declare i8 @unary_mod1_Integer8(i8)

declare i8 @checked_unary_mod1_Integer8(i8)

declare i16 @unary_mod1_Integer16(i16)

declare i16 @checked_unary_mod1_Integer16(i16)

declare i32 @unary_mod1_Integer32(i32)

declare i32 @checked_unary_mod1_Integer32(i32)

declare i64 @unary_mod1_Integer64(i64)

declare i64 @checked_unary_mod1_Integer64(i64)

declare i8 @unary_mod1_UnsignedInteger8(i8)

declare i8 @checked_unary_mod1_UnsignedInteger8(i8)

declare i16 @unary_mod1_UnsignedInteger16(i16)

declare i16 @checked_unary_mod1_UnsignedInteger16(i16)

declare i32 @unary_mod1_UnsignedInteger32(i32)

declare i32 @checked_unary_mod1_UnsignedInteger32(i32)

declare i64 @unary_mod1_UnsignedInteger64(i64)

declare i64 @checked_unary_mod1_UnsignedInteger64(i64)

declare i16 @unary_mod1_Real16(i16)

declare i16 @checked_unary_mod1_Real16(i16)

declare float @unary_mod1_Real32(float)

declare float @checked_unary_mod1_Real32(float)

declare double @unary_mod1_Real64(double)

declare double @checked_unary_mod1_Real64(double)

declare void @unary_mod1_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_mod1_ComplexReal32(float*, float*, float, float)

declare void @unary_mod1_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_mod1_ComplexReal64(double*, double*, double, double)

declare i64 @unary_neg_Integer8(i8)

declare i64 @checked_unary_neg_Integer8(i8)

declare i64 @unary_neg_Integer16(i16)

declare i64 @checked_unary_neg_Integer16(i16)

declare i64 @unary_neg_Integer32(i32)

declare i64 @checked_unary_neg_Integer32(i32)

declare i64 @unary_neg_Integer64(i64)

declare i64 @checked_unary_neg_Integer64(i64)

declare i64 @unary_neg_UnsignedInteger8(i8)

declare i64 @checked_unary_neg_UnsignedInteger8(i8)

declare i64 @unary_neg_UnsignedInteger16(i16)

declare i64 @checked_unary_neg_UnsignedInteger16(i16)

declare i64 @unary_neg_UnsignedInteger32(i32)

declare i64 @checked_unary_neg_UnsignedInteger32(i32)

declare i64 @unary_neg_UnsignedInteger64(i64)

declare i64 @checked_unary_neg_UnsignedInteger64(i64)

declare i64 @unary_neg_Real16(i16)

declare i64 @checked_unary_neg_Real16(i16)

declare i64 @unary_neg_Real32(float)

declare i64 @checked_unary_neg_Real32(float)

declare i64 @unary_neg_Real64(double)

declare i64 @checked_unary_neg_Real64(double)

declare i64 @unary_neg_ComplexReal32(<2 x float>)

declare i64 @checked_unary_neg_ComplexReal32(<2 x float>)

declare i64 @unary_neg_ComplexReal64(double, double)

declare i64 @checked_unary_neg_ComplexReal64(double, double)

declare i64 @unary_nneg_Integer8(i8)

declare i64 @checked_unary_nneg_Integer8(i8)

declare i64 @unary_nneg_Integer16(i16)

declare i64 @checked_unary_nneg_Integer16(i16)

declare i64 @unary_nneg_Integer32(i32)

declare i64 @checked_unary_nneg_Integer32(i32)

declare i64 @unary_nneg_Integer64(i64)

declare i64 @checked_unary_nneg_Integer64(i64)

declare i64 @unary_nneg_UnsignedInteger8(i8)

declare i64 @checked_unary_nneg_UnsignedInteger8(i8)

declare i64 @unary_nneg_UnsignedInteger16(i16)

declare i64 @checked_unary_nneg_UnsignedInteger16(i16)

declare i64 @unary_nneg_UnsignedInteger32(i32)

declare i64 @checked_unary_nneg_UnsignedInteger32(i32)

declare i64 @unary_nneg_UnsignedInteger64(i64)

declare i64 @checked_unary_nneg_UnsignedInteger64(i64)

declare i64 @unary_nneg_Real16(i16)

declare i64 @checked_unary_nneg_Real16(i16)

declare i64 @unary_nneg_Real32(float)

declare i64 @checked_unary_nneg_Real32(float)

declare i64 @unary_nneg_Real64(double)

declare i64 @checked_unary_nneg_Real64(double)

declare i64 @unary_nneg_ComplexReal32(<2 x float>)

declare i64 @checked_unary_nneg_ComplexReal32(<2 x float>)

declare i64 @unary_nneg_ComplexReal64(double, double)

declare i64 @checked_unary_nneg_ComplexReal64(double, double)

declare i64 @unary_npos_Integer8(i8)

declare i64 @checked_unary_npos_Integer8(i8)

declare i64 @unary_npos_Integer16(i16)

declare i64 @checked_unary_npos_Integer16(i16)

declare i64 @unary_npos_Integer32(i32)

declare i64 @checked_unary_npos_Integer32(i32)

declare i64 @unary_npos_Integer64(i64)

declare i64 @checked_unary_npos_Integer64(i64)

declare i64 @unary_npos_UnsignedInteger8(i8)

declare i64 @checked_unary_npos_UnsignedInteger8(i8)

declare i64 @unary_npos_UnsignedInteger16(i16)

declare i64 @checked_unary_npos_UnsignedInteger16(i16)

declare i64 @unary_npos_UnsignedInteger32(i32)

declare i64 @checked_unary_npos_UnsignedInteger32(i32)

declare i64 @unary_npos_UnsignedInteger64(i64)

declare i64 @checked_unary_npos_UnsignedInteger64(i64)

declare i64 @unary_npos_Real16(i16)

declare i64 @checked_unary_npos_Real16(i16)

declare i64 @unary_npos_Real32(float)

declare i64 @checked_unary_npos_Real32(float)

declare i64 @unary_npos_Real64(double)

declare i64 @checked_unary_npos_Real64(double)

declare i64 @unary_npos_ComplexReal32(<2 x float>)

declare i64 @checked_unary_npos_ComplexReal32(<2 x float>)

declare i64 @unary_npos_ComplexReal64(double, double)

declare i64 @checked_unary_npos_ComplexReal64(double, double)

declare i64 @unary_oddq_Integer8(i8)

declare i64 @checked_unary_oddq_Integer8(i8)

declare i64 @unary_oddq_Integer16(i16)

declare i64 @checked_unary_oddq_Integer16(i16)

declare i64 @unary_oddq_Integer32(i32)

declare i64 @checked_unary_oddq_Integer32(i32)

declare i64 @unary_oddq_Integer64(i64)

declare i64 @checked_unary_oddq_Integer64(i64)

declare i64 @unary_oddq_UnsignedInteger8(i8)

declare i64 @checked_unary_oddq_UnsignedInteger8(i8)

declare i64 @unary_oddq_UnsignedInteger16(i16)

declare i64 @checked_unary_oddq_UnsignedInteger16(i16)

declare i64 @unary_oddq_UnsignedInteger32(i32)

declare i64 @checked_unary_oddq_UnsignedInteger32(i32)

declare i64 @unary_oddq_UnsignedInteger64(i64)

declare i64 @checked_unary_oddq_UnsignedInteger64(i64)

declare i64 @unary_oddq_Real16(i16)

declare i64 @checked_unary_oddq_Real16(i16)

declare i64 @unary_oddq_Real32(float)

declare i64 @checked_unary_oddq_Real32(float)

declare i64 @unary_oddq_Real64(double)

declare i64 @checked_unary_oddq_Real64(double)

declare i64 @unary_oddq_ComplexReal32(<2 x float>)

declare i64 @checked_unary_oddq_ComplexReal32(<2 x float>)

declare i64 @unary_oddq_ComplexReal64(double, double)

declare i64 @checked_unary_oddq_ComplexReal64(double, double)

declare i8 @unary_one_Integer8(i8)

declare i8 @checked_unary_one_Integer8(i8)

declare i16 @unary_one_Integer16(i16)

declare i16 @checked_unary_one_Integer16(i16)

declare i32 @unary_one_Integer32(i32)

declare i32 @checked_unary_one_Integer32(i32)

declare i64 @unary_one_Integer64(i64)

declare i64 @checked_unary_one_Integer64(i64)

declare i8 @unary_one_UnsignedInteger8(i8)

declare i8 @checked_unary_one_UnsignedInteger8(i8)

declare i16 @unary_one_UnsignedInteger16(i16)

declare i16 @checked_unary_one_UnsignedInteger16(i16)

declare i32 @unary_one_UnsignedInteger32(i32)

declare i32 @checked_unary_one_UnsignedInteger32(i32)

declare i64 @unary_one_UnsignedInteger64(i64)

declare i64 @checked_unary_one_UnsignedInteger64(i64)

declare i16 @unary_one_Real16(i16)

declare i16 @checked_unary_one_Real16(i16)

declare float @unary_one_Real32(float)

declare float @checked_unary_one_Real32(float)

declare double @unary_one_Real64(double)

declare double @checked_unary_one_Real64(double)

declare void @unary_one_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_one_ComplexReal32(float*, float*, float, float)

declare void @unary_one_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_one_ComplexReal64(double*, double*, double, double)

declare i64 @unary_pos_Integer8(i8)

declare i64 @checked_unary_pos_Integer8(i8)

declare i64 @unary_pos_Integer16(i16)

declare i64 @checked_unary_pos_Integer16(i16)

declare i64 @unary_pos_Integer32(i32)

declare i64 @checked_unary_pos_Integer32(i32)

declare i64 @unary_pos_Integer64(i64)

declare i64 @checked_unary_pos_Integer64(i64)

declare i64 @unary_pos_UnsignedInteger8(i8)

declare i64 @checked_unary_pos_UnsignedInteger8(i8)

declare i64 @unary_pos_UnsignedInteger16(i16)

declare i64 @checked_unary_pos_UnsignedInteger16(i16)

declare i64 @unary_pos_UnsignedInteger32(i32)

declare i64 @checked_unary_pos_UnsignedInteger32(i32)

declare i64 @unary_pos_UnsignedInteger64(i64)

declare i64 @checked_unary_pos_UnsignedInteger64(i64)

declare i64 @unary_pos_Real16(i16)

declare i64 @checked_unary_pos_Real16(i16)

declare i64 @unary_pos_Real32(float)

declare i64 @checked_unary_pos_Real32(float)

declare i64 @unary_pos_Real64(double)

declare i64 @checked_unary_pos_Real64(double)

declare i64 @unary_pos_ComplexReal32(<2 x float>)

declare i64 @checked_unary_pos_ComplexReal32(<2 x float>)

declare i64 @unary_pos_ComplexReal64(double, double)

declare i64 @checked_unary_pos_ComplexReal64(double, double)

declare i8 @unary_ramp_Integer8(i8)

declare i8 @checked_unary_ramp_Integer8(i8)

declare i16 @unary_ramp_Integer16(i16)

declare i16 @checked_unary_ramp_Integer16(i16)

declare i32 @unary_ramp_Integer32(i32)

declare i32 @checked_unary_ramp_Integer32(i32)

declare i64 @unary_ramp_Integer64(i64)

declare i64 @checked_unary_ramp_Integer64(i64)

declare i8 @unary_ramp_UnsignedInteger8(i8)

declare i8 @checked_unary_ramp_UnsignedInteger8(i8)

declare i16 @unary_ramp_UnsignedInteger16(i16)

declare i16 @checked_unary_ramp_UnsignedInteger16(i16)

declare i32 @unary_ramp_UnsignedInteger32(i32)

declare i32 @checked_unary_ramp_UnsignedInteger32(i32)

declare i64 @unary_ramp_UnsignedInteger64(i64)

declare i64 @checked_unary_ramp_UnsignedInteger64(i64)

declare i16 @unary_ramp_Real16(i16)

declare i16 @checked_unary_ramp_Real16(i16)

declare float @unary_ramp_Real32(float)

declare float @checked_unary_ramp_Real32(float)

declare double @unary_ramp_Real64(double)

declare double @checked_unary_ramp_Real64(double)

declare void @unary_ramp_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_ramp_ComplexReal32(float*, float*, float, float)

declare void @unary_ramp_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_ramp_ComplexReal64(double*, double*, double, double)

declare i8 @unary_re_Integer8(i8)

declare i8 @checked_unary_re_Integer8(i8)

declare i16 @unary_re_Integer16(i16)

declare i16 @checked_unary_re_Integer16(i16)

declare i32 @unary_re_Integer32(i32)

declare i32 @checked_unary_re_Integer32(i32)

declare i64 @unary_re_Integer64(i64)

declare i64 @checked_unary_re_Integer64(i64)

declare i8 @unary_re_UnsignedInteger8(i8)

declare i8 @checked_unary_re_UnsignedInteger8(i8)

declare i16 @unary_re_UnsignedInteger16(i16)

declare i16 @checked_unary_re_UnsignedInteger16(i16)

declare i32 @unary_re_UnsignedInteger32(i32)

declare i32 @checked_unary_re_UnsignedInteger32(i32)

declare i64 @unary_re_UnsignedInteger64(i64)

declare i64 @checked_unary_re_UnsignedInteger64(i64)

declare i16 @unary_re_Real16(i16)

declare i16 @checked_unary_re_Real16(i16)

declare float @unary_re_Real32(float)

declare float @checked_unary_re_Real32(float)

declare double @unary_re_Real64(double)

declare double @checked_unary_re_Real64(double)

declare float @unary_re_ComplexReal32(<2 x float>)

declare float @checked_unary_re_ComplexReal32(<2 x float>)

declare double @unary_re_ComplexReal64(double, double)

declare double @checked_unary_re_ComplexReal64(double, double)

declare i8 @unary_recip_Integer8(i8)

declare i8 @checked_unary_recip_Integer8(i8)

declare i16 @unary_recip_Integer16(i16)

declare i16 @checked_unary_recip_Integer16(i16)

declare i32 @unary_recip_Integer32(i32)

declare i32 @checked_unary_recip_Integer32(i32)

declare i64 @unary_recip_Integer64(i64)

declare i64 @checked_unary_recip_Integer64(i64)

declare i8 @unary_recip_UnsignedInteger8(i8)

declare i8 @checked_unary_recip_UnsignedInteger8(i8)

declare i16 @unary_recip_UnsignedInteger16(i16)

declare i16 @checked_unary_recip_UnsignedInteger16(i16)

declare i32 @unary_recip_UnsignedInteger32(i32)

declare i32 @checked_unary_recip_UnsignedInteger32(i32)

declare i64 @unary_recip_UnsignedInteger64(i64)

declare i64 @checked_unary_recip_UnsignedInteger64(i64)

declare i16 @unary_recip_Real16(i16)

declare i16 @checked_unary_recip_Real16(i16)

declare float @unary_recip_Real32(float)

declare float @checked_unary_recip_Real32(float)

declare double @unary_recip_Real64(double)

declare double @checked_unary_recip_Real64(double)

declare void @unary_recip_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_recip_ComplexReal32(float*, float*, float, float)

declare <2 x float> @_ZN3wrt9scalar_op6binaryL9divide_opINSt3__17complexIfEES5_LNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE9EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE.219(%"class.std::__1::complex.156"*, %"class.std::__1::complex.156"*)

declare void @unary_recip_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_recip_ComplexReal64(double*, double*, double, double)

declare { double, double } @_ZN3wrt9scalar_op6binaryL9divide_opINSt3__17complexIdEES5_LNS_13runtime_flagsE1EEENS3_9enable_ifIXoosr28is_floating_point_or_complexIT_EE5valuesr28is_floating_point_or_complexIT0_EE5valueENS1_6detail7op_infoILNSA_2opE9EN7rt_typeIS8_E4typeENSD_IS9_E4typeEE11result_typeEE4typeERKNSI_19first_argument_typeERKNSI_20second_argument_typeE.220(%"class.std::__1::complex"*, %"class.std::__1::complex"*)

declare i8 @unary_round_Integer8(i8)

declare i8 @checked_unary_round_Integer8(i8)

declare i16 @unary_round_Integer16(i16)

declare i16 @checked_unary_round_Integer16(i16)

declare i32 @unary_round_Integer32(i32)

declare i32 @checked_unary_round_Integer32(i32)

declare i64 @unary_round_Integer64(i64)

declare i64 @checked_unary_round_Integer64(i64)

declare i8 @unary_round_UnsignedInteger8(i8)

declare i8 @checked_unary_round_UnsignedInteger8(i8)

declare i16 @unary_round_UnsignedInteger16(i16)

declare i16 @checked_unary_round_UnsignedInteger16(i16)

declare i32 @unary_round_UnsignedInteger32(i32)

declare i32 @checked_unary_round_UnsignedInteger32(i32)

declare i64 @unary_round_UnsignedInteger64(i64)

declare i64 @checked_unary_round_UnsignedInteger64(i64)

declare i64 @unary_round_Real16(i16)

declare i64 @checked_unary_round_Real16(i16)

declare i64 @unary_round_Real32(float)

declare i64 @checked_unary_round_Real32(float)

declare i64 @unary_round_Real64(double)

declare i64 @checked_unary_round_Real64(double)

declare i64 @unary_round_ComplexReal32(<2 x float>)

declare i64 @checked_unary_round_ComplexReal32(<2 x float>)

declare i64 @unary_round_ComplexReal64(double, double)

declare i64 @checked_unary_round_ComplexReal64(double, double)

declare i8 @unary_rsqrt_Integer8(i8)

declare i8 @checked_unary_rsqrt_Integer8(i8)

declare i16 @unary_rsqrt_Integer16(i16)

declare i16 @checked_unary_rsqrt_Integer16(i16)

declare i32 @unary_rsqrt_Integer32(i32)

declare i32 @checked_unary_rsqrt_Integer32(i32)

declare i64 @unary_rsqrt_Integer64(i64)

declare i64 @checked_unary_rsqrt_Integer64(i64)

declare i8 @unary_rsqrt_UnsignedInteger8(i8)

declare i8 @checked_unary_rsqrt_UnsignedInteger8(i8)

declare i16 @unary_rsqrt_UnsignedInteger16(i16)

declare i16 @checked_unary_rsqrt_UnsignedInteger16(i16)

declare i32 @unary_rsqrt_UnsignedInteger32(i32)

declare i32 @checked_unary_rsqrt_UnsignedInteger32(i32)

declare i64 @unary_rsqrt_UnsignedInteger64(i64)

declare i64 @checked_unary_rsqrt_UnsignedInteger64(i64)

declare i16 @unary_rsqrt_Real16(i16)

declare i16 @checked_unary_rsqrt_Real16(i16)

declare float @unary_rsqrt_Real32(float)

declare float @checked_unary_rsqrt_Real32(float)

declare double @unary_rsqrt_Real64(double)

declare double @checked_unary_rsqrt_Real64(double)

declare void @unary_rsqrt_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_rsqrt_ComplexReal32(float*, float*, float, float)

declare void @unary_rsqrt_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_rsqrt_ComplexReal64(double*, double*, double, double)

declare i8 @unary_sec_Integer8(i8)

declare i8 @checked_unary_sec_Integer8(i8)

declare i16 @unary_sec_Integer16(i16)

declare i16 @checked_unary_sec_Integer16(i16)

declare i32 @unary_sec_Integer32(i32)

declare i32 @checked_unary_sec_Integer32(i32)

declare i64 @unary_sec_Integer64(i64)

declare i64 @checked_unary_sec_Integer64(i64)

declare i8 @unary_sec_UnsignedInteger8(i8)

declare i8 @checked_unary_sec_UnsignedInteger8(i8)

declare i16 @unary_sec_UnsignedInteger16(i16)

declare i16 @checked_unary_sec_UnsignedInteger16(i16)

declare i32 @unary_sec_UnsignedInteger32(i32)

declare i32 @checked_unary_sec_UnsignedInteger32(i32)

declare i64 @unary_sec_UnsignedInteger64(i64)

declare i64 @checked_unary_sec_UnsignedInteger64(i64)

declare i16 @unary_sec_Real16(i16)

declare i16 @checked_unary_sec_Real16(i16)

declare float @unary_sec_Real32(float)

declare float @checked_unary_sec_Real32(float)

declare double @unary_sec_Real64(double)

declare double @checked_unary_sec_Real64(double)

declare void @unary_sec_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_sec_ComplexReal32(float*, float*, float, float)

declare void @unary_sec_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_sec_ComplexReal64(double*, double*, double, double)

declare i8 @unary_sech_Integer8(i8)

declare i8 @checked_unary_sech_Integer8(i8)

declare i16 @unary_sech_Integer16(i16)

declare i16 @checked_unary_sech_Integer16(i16)

declare i32 @unary_sech_Integer32(i32)

declare i32 @checked_unary_sech_Integer32(i32)

declare i64 @unary_sech_Integer64(i64)

declare i64 @checked_unary_sech_Integer64(i64)

declare i8 @unary_sech_UnsignedInteger8(i8)

declare i8 @checked_unary_sech_UnsignedInteger8(i8)

declare i16 @unary_sech_UnsignedInteger16(i16)

declare i16 @checked_unary_sech_UnsignedInteger16(i16)

declare i32 @unary_sech_UnsignedInteger32(i32)

declare i32 @checked_unary_sech_UnsignedInteger32(i32)

declare i64 @unary_sech_UnsignedInteger64(i64)

declare i64 @checked_unary_sech_UnsignedInteger64(i64)

declare i16 @unary_sech_Real16(i16)

declare i16 @checked_unary_sech_Real16(i16)

declare float @unary_sech_Real32(float)

declare float @checked_unary_sech_Real32(float)

declare double @unary_sech_Real64(double)

declare double @checked_unary_sech_Real64(double)

declare void @unary_sech_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_sech_ComplexReal32(float*, float*, float, float)

declare void @unary_sech_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_sech_ComplexReal64(double*, double*, double, double)

declare i8 @unary_sign_Integer8(i8)

declare i8 @checked_unary_sign_Integer8(i8)

declare i16 @unary_sign_Integer16(i16)

declare i16 @checked_unary_sign_Integer16(i16)

declare i32 @unary_sign_Integer32(i32)

declare i32 @checked_unary_sign_Integer32(i32)

declare i64 @unary_sign_Integer64(i64)

declare i64 @checked_unary_sign_Integer64(i64)

declare i8 @unary_sign_UnsignedInteger8(i8)

declare i8 @checked_unary_sign_UnsignedInteger8(i8)

declare i16 @unary_sign_UnsignedInteger16(i16)

declare i16 @checked_unary_sign_UnsignedInteger16(i16)

declare i32 @unary_sign_UnsignedInteger32(i32)

declare i32 @checked_unary_sign_UnsignedInteger32(i32)

declare i64 @unary_sign_UnsignedInteger64(i64)

declare i64 @checked_unary_sign_UnsignedInteger64(i64)

declare i64 @unary_sign_Real16(i16)

declare i64 @checked_unary_sign_Real16(i16)

declare i64 @unary_sign_Real32(float)

declare i64 @checked_unary_sign_Real32(float)

declare i64 @unary_sign_Real64(double)

declare i64 @checked_unary_sign_Real64(double)

declare void @unary_sign_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_sign_ComplexReal32(float*, float*, float, float)

declare void @unary_sign_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_sign_ComplexReal64(double*, double*, double, double)

declare i8 @unary_sin_Integer8(i8)

declare i8 @checked_unary_sin_Integer8(i8)

declare i16 @unary_sin_Integer16(i16)

declare i16 @checked_unary_sin_Integer16(i16)

declare i32 @unary_sin_Integer32(i32)

declare i32 @checked_unary_sin_Integer32(i32)

declare i64 @unary_sin_Integer64(i64)

declare i64 @checked_unary_sin_Integer64(i64)

declare i8 @unary_sin_UnsignedInteger8(i8)

declare i8 @checked_unary_sin_UnsignedInteger8(i8)

declare i16 @unary_sin_UnsignedInteger16(i16)

declare i16 @checked_unary_sin_UnsignedInteger16(i16)

declare i32 @unary_sin_UnsignedInteger32(i32)

declare i32 @checked_unary_sin_UnsignedInteger32(i32)

declare i64 @unary_sin_UnsignedInteger64(i64)

declare i64 @checked_unary_sin_UnsignedInteger64(i64)

declare i16 @unary_sin_Real16(i16)

declare i16 @checked_unary_sin_Real16(i16)

declare float @unary_sin_Real32(float)

declare float @checked_unary_sin_Real32(float)

declare double @unary_sin_Real64(double)

declare double @checked_unary_sin_Real64(double)

declare void @unary_sin_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_sin_ComplexReal32(float*, float*, float, float)

declare void @unary_sin_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_sin_ComplexReal64(double*, double*, double, double)

declare i8 @unary_sinc_Integer8(i8)

declare i8 @checked_unary_sinc_Integer8(i8)

declare i16 @unary_sinc_Integer16(i16)

declare i16 @checked_unary_sinc_Integer16(i16)

declare i32 @unary_sinc_Integer32(i32)

declare i32 @checked_unary_sinc_Integer32(i32)

declare i64 @unary_sinc_Integer64(i64)

declare i64 @checked_unary_sinc_Integer64(i64)

declare i8 @unary_sinc_UnsignedInteger8(i8)

declare i8 @checked_unary_sinc_UnsignedInteger8(i8)

declare i16 @unary_sinc_UnsignedInteger16(i16)

declare i16 @checked_unary_sinc_UnsignedInteger16(i16)

declare i32 @unary_sinc_UnsignedInteger32(i32)

declare i32 @checked_unary_sinc_UnsignedInteger32(i32)

declare i64 @unary_sinc_UnsignedInteger64(i64)

declare i64 @checked_unary_sinc_UnsignedInteger64(i64)

declare i16 @unary_sinc_Real16(i16)

declare i16 @checked_unary_sinc_Real16(i16)

declare float @unary_sinc_Real32(float)

declare float @checked_unary_sinc_Real32(float)

declare double @unary_sinc_Real64(double)

declare double @checked_unary_sinc_Real64(double)

declare void @unary_sinc_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_sinc_ComplexReal32(float*, float*, float, float)

declare void @unary_sinc_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_sinc_ComplexReal64(double*, double*, double, double)

declare i8 @unary_sinh_Integer8(i8)

declare i8 @checked_unary_sinh_Integer8(i8)

declare i16 @unary_sinh_Integer16(i16)

declare i16 @checked_unary_sinh_Integer16(i16)

declare i32 @unary_sinh_Integer32(i32)

declare i32 @checked_unary_sinh_Integer32(i32)

declare i64 @unary_sinh_Integer64(i64)

declare i64 @checked_unary_sinh_Integer64(i64)

declare i8 @unary_sinh_UnsignedInteger8(i8)

declare i8 @checked_unary_sinh_UnsignedInteger8(i8)

declare i16 @unary_sinh_UnsignedInteger16(i16)

declare i16 @checked_unary_sinh_UnsignedInteger16(i16)

declare i32 @unary_sinh_UnsignedInteger32(i32)

declare i32 @checked_unary_sinh_UnsignedInteger32(i32)

declare i64 @unary_sinh_UnsignedInteger64(i64)

declare i64 @checked_unary_sinh_UnsignedInteger64(i64)

declare i16 @unary_sinh_Real16(i16)

declare i16 @checked_unary_sinh_Real16(i16)

declare float @unary_sinh_Real32(float)

declare float @checked_unary_sinh_Real32(float)

declare double @unary_sinh_Real64(double)

declare double @checked_unary_sinh_Real64(double)

declare void @unary_sinh_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_sinh_ComplexReal32(float*, float*, float, float)

declare void @unary_sinh_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_sinh_ComplexReal64(double*, double*, double, double)

declare i8 @unary_sqrt_Integer8(i8)

declare i8 @checked_unary_sqrt_Integer8(i8)

declare i16 @unary_sqrt_Integer16(i16)

declare i16 @checked_unary_sqrt_Integer16(i16)

declare i32 @unary_sqrt_Integer32(i32)

declare i32 @checked_unary_sqrt_Integer32(i32)

declare i64 @unary_sqrt_Integer64(i64)

declare i64 @checked_unary_sqrt_Integer64(i64)

declare i8 @unary_sqrt_UnsignedInteger8(i8)

declare i8 @checked_unary_sqrt_UnsignedInteger8(i8)

declare i16 @unary_sqrt_UnsignedInteger16(i16)

declare i16 @checked_unary_sqrt_UnsignedInteger16(i16)

declare i32 @unary_sqrt_UnsignedInteger32(i32)

declare i32 @checked_unary_sqrt_UnsignedInteger32(i32)

declare i64 @unary_sqrt_UnsignedInteger64(i64)

declare i64 @checked_unary_sqrt_UnsignedInteger64(i64)

declare i16 @unary_sqrt_Real16(i16)

declare i16 @checked_unary_sqrt_Real16(i16)

declare float @unary_sqrt_Real32(float)

declare float @checked_unary_sqrt_Real32(float)

declare double @unary_sqrt_Real64(double)

declare double @checked_unary_sqrt_Real64(double)

declare void @unary_sqrt_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_sqrt_ComplexReal32(float*, float*, float, float)

declare void @unary_sqrt_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_sqrt_ComplexReal64(double*, double*, double, double)

declare i8 @unary_square_Integer8(i8)

declare i8 @checked_unary_square_Integer8(i8)

declare i16 @unary_square_Integer16(i16)

declare i16 @checked_unary_square_Integer16(i16)

declare i32 @unary_square_Integer32(i32)

declare i32 @checked_unary_square_Integer32(i32)

declare i64 @unary_square_Integer64(i64)

declare i64 @checked_unary_square_Integer64(i64)

declare i8 @unary_square_UnsignedInteger8(i8)

declare i8 @checked_unary_square_UnsignedInteger8(i8)

declare i16 @unary_square_UnsignedInteger16(i16)

declare i16 @checked_unary_square_UnsignedInteger16(i16)

declare i32 @unary_square_UnsignedInteger32(i32)

declare i32 @checked_unary_square_UnsignedInteger32(i32)

declare i64 @unary_square_UnsignedInteger64(i64)

declare i64 @checked_unary_square_UnsignedInteger64(i64)

declare i16 @unary_square_Real16(i16)

declare i16 @checked_unary_square_Real16(i16)

declare float @unary_square_Real32(float)

declare float @checked_unary_square_Real32(float)

declare double @unary_square_Real64(double)

declare double @checked_unary_square_Real64(double)

declare void @unary_square_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_square_ComplexReal32(float*, float*, float, float)

declare void @unary_square_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_square_ComplexReal64(double*, double*, double, double)

declare i8 @unary_tan_Integer8(i8)

declare i8 @checked_unary_tan_Integer8(i8)

declare i16 @unary_tan_Integer16(i16)

declare i16 @checked_unary_tan_Integer16(i16)

declare i32 @unary_tan_Integer32(i32)

declare i32 @checked_unary_tan_Integer32(i32)

declare i64 @unary_tan_Integer64(i64)

declare i64 @checked_unary_tan_Integer64(i64)

declare i8 @unary_tan_UnsignedInteger8(i8)

declare i8 @checked_unary_tan_UnsignedInteger8(i8)

declare i16 @unary_tan_UnsignedInteger16(i16)

declare i16 @checked_unary_tan_UnsignedInteger16(i16)

declare i32 @unary_tan_UnsignedInteger32(i32)

declare i32 @checked_unary_tan_UnsignedInteger32(i32)

declare i64 @unary_tan_UnsignedInteger64(i64)

declare i64 @checked_unary_tan_UnsignedInteger64(i64)

declare i16 @unary_tan_Real16(i16)

declare i16 @checked_unary_tan_Real16(i16)

declare float @unary_tan_Real32(float)

declare float @checked_unary_tan_Real32(float)

declare double @unary_tan_Real64(double)

declare double @checked_unary_tan_Real64(double)

declare void @unary_tan_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_tan_ComplexReal32(float*, float*, float, float)

declare void @unary_tan_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_tan_ComplexReal64(double*, double*, double, double)

declare i8 @unary_tanh_Integer8(i8)

declare i8 @checked_unary_tanh_Integer8(i8)

declare i16 @unary_tanh_Integer16(i16)

declare i16 @checked_unary_tanh_Integer16(i16)

declare i32 @unary_tanh_Integer32(i32)

declare i32 @checked_unary_tanh_Integer32(i32)

declare i64 @unary_tanh_Integer64(i64)

declare i64 @checked_unary_tanh_Integer64(i64)

declare i8 @unary_tanh_UnsignedInteger8(i8)

declare i8 @checked_unary_tanh_UnsignedInteger8(i8)

declare i16 @unary_tanh_UnsignedInteger16(i16)

declare i16 @checked_unary_tanh_UnsignedInteger16(i16)

declare i32 @unary_tanh_UnsignedInteger32(i32)

declare i32 @checked_unary_tanh_UnsignedInteger32(i32)

declare i64 @unary_tanh_UnsignedInteger64(i64)

declare i64 @checked_unary_tanh_UnsignedInteger64(i64)

declare i16 @unary_tanh_Real16(i16)

declare i16 @checked_unary_tanh_Real16(i16)

declare float @unary_tanh_Real32(float)

declare float @checked_unary_tanh_Real32(float)

declare double @unary_tanh_Real64(double)

declare double @checked_unary_tanh_Real64(double)

declare void @unary_tanh_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_tanh_ComplexReal32(float*, float*, float, float)

declare void @unary_tanh_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_tanh_ComplexReal64(double*, double*, double, double)

declare i8 @unary_unitize_Integer8(i8)

declare i8 @checked_unary_unitize_Integer8(i8)

declare i16 @unary_unitize_Integer16(i16)

declare i16 @checked_unary_unitize_Integer16(i16)

declare i32 @unary_unitize_Integer32(i32)

declare i32 @checked_unary_unitize_Integer32(i32)

declare i64 @unary_unitize_Integer64(i64)

declare i64 @checked_unary_unitize_Integer64(i64)

declare i8 @unary_unitize_UnsignedInteger8(i8)

declare i8 @checked_unary_unitize_UnsignedInteger8(i8)

declare i16 @unary_unitize_UnsignedInteger16(i16)

declare i16 @checked_unary_unitize_UnsignedInteger16(i16)

declare i32 @unary_unitize_UnsignedInteger32(i32)

declare i32 @checked_unary_unitize_UnsignedInteger32(i32)

declare i64 @unary_unitize_UnsignedInteger64(i64)

declare i64 @checked_unary_unitize_UnsignedInteger64(i64)

declare i64 @unary_unitize_Real16(i16)

declare i64 @checked_unary_unitize_Real16(i16)

declare i64 @unary_unitize_Real32(float)

declare i64 @checked_unary_unitize_Real32(float)

declare i64 @unary_unitize_Real64(double)

declare i64 @checked_unary_unitize_Real64(double)

declare i64 @unary_unitize_ComplexReal32(<2 x float>)

declare i64 @checked_unary_unitize_ComplexReal32(<2 x float>)

declare i64 @unary_unitize_ComplexReal64(double, double)

declare i64 @checked_unary_unitize_ComplexReal64(double, double)

declare i8 @unary_zero_Integer8(i8)

declare i8 @checked_unary_zero_Integer8(i8)

declare i16 @unary_zero_Integer16(i16)

declare i16 @checked_unary_zero_Integer16(i16)

declare i32 @unary_zero_Integer32(i32)

declare i32 @checked_unary_zero_Integer32(i32)

declare i64 @unary_zero_Integer64(i64)

declare i64 @checked_unary_zero_Integer64(i64)

declare i8 @unary_zero_UnsignedInteger8(i8)

declare i8 @checked_unary_zero_UnsignedInteger8(i8)

declare i16 @unary_zero_UnsignedInteger16(i16)

declare i16 @checked_unary_zero_UnsignedInteger16(i16)

declare i32 @unary_zero_UnsignedInteger32(i32)

declare i32 @checked_unary_zero_UnsignedInteger32(i32)

declare i64 @unary_zero_UnsignedInteger64(i64)

declare i64 @checked_unary_zero_UnsignedInteger64(i64)

declare i16 @unary_zero_Real16(i16)

declare i16 @checked_unary_zero_Real16(i16)

declare float @unary_zero_Real32(float)

declare float @checked_unary_zero_Real32(float)

declare double @unary_zero_Real64(double)

declare double @checked_unary_zero_Real64(double)

declare void @unary_zero_ComplexReal32(float*, float*, float, float)

declare void @checked_unary_zero_ComplexReal32(float*, float*, float, float)

declare void @unary_zero_ComplexReal64(double*, double*, double, double)

declare void @checked_unary_zero_ComplexReal64(double*, double*, double, double)

declare i8* @to_string_Integer8(i8)

declare i8* @strcpy(i8*, i8*)

declare i8* @to_string_Integer16(i16)

declare i8* @to_string_Integer32(i32)

declare i8* @to_string_Integer64(i64)

declare void @_ZNSt3__19to_stringEx(%"class.std::__1::basic_string"*, i64)

declare i8* @to_string_UnsignedInteger8(i8)

declare i8* @to_string_UnsignedInteger16(i16)

declare i8* @to_string_UnsignedInteger32(i32)

declare void @_ZNSt3__19to_stringEj(%"class.std::__1::basic_string"*, i32)

declare i8* @to_string_UnsignedInteger64(i64)

declare void @_ZNSt3__19to_stringEy(%"class.std::__1::basic_string"*, i64)

declare i8* @to_string_Real16(i16)

declare void @_ZNSt3__19to_stringEf(%"class.std::__1::basic_string"*, float)

declare i8* @to_string_Real32(float)

declare i8* @to_string_Real64(double)

declare void @_ZNSt3__19to_stringEd(%"class.std::__1::basic_string"*, double)

declare i8* @to_string_ComplexReal32(<2 x float>)

declare %"class.std::__1::basic_string"* @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6appendEPKc(%"class.std::__1::basic_string"*, i8*)

declare %"class.std::__1::basic_string"* @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6appendEPKcm(%"class.std::__1::basic_string"*, i8*, i64)

declare i8* @to_string_ComplexReal64(double, double)

declare void @to_string_free(i8*)

declare i32 @parallel_for_Integer64_Integer64(i64*, i64, i64*, i64 (i64)*)

declare i32 @_ZNSt3__16thread20hardware_concurrencyEv()

declare void @_ZNSt3__16vectorINS_6futureIvEENS_9allocatorIS2_EEE7reserveEm(%"class.std::__1::vector"*, i64)

declare void @_ZNSt3__115__thread_structC1Ev(%"class.std::__1::__thread_struct"*)

declare i8* @"_ZNSt3__114__thread_proxyINS_5tupleIJNS_10unique_ptrINS_15__thread_structENS_14default_deleteIS3_EEEEMNS_19__async_assoc_stateIvNS_12__async_funcIZN5utilsL12parallel_forINS9_16execution_policyIvEEPxSD_Z32parallel_for_Integer64_Integer64E3$_0EE5errorRKT_T0_SJ_T1_SK_OT2_EUlvE_JEEEEEFvvEPSP_EEEEEPvSU_"(i8*)

declare i32 @pthread_create(%struct._opaque_pthread_t**, %struct._opaque_pthread_mutex_t*, i8* (i8*)*, i8*)

declare void @_ZNSt3__115__thread_structD1Ev(%"class.std::__1::__thread_struct"*)

declare void @_ZNSt3__120__throw_system_errorEiPKc(i32, i8*)

declare void @_ZNSt3__16thread6detachEv(%"class.std::__1::thread"*)

declare void @_ZNSt3__16threadD1Ev(%"class.std::__1::thread"*)

declare void @_ZNSt3__16futureIvEC1EPNS_17__assoc_sub_stateE(%"class.std::__1::future"*, %"class.std::__1::__assoc_sub_state"*)

declare void @__cxa_rethrow()

declare void @_ZNSt3__16vectorINS_6futureIvEENS_9allocatorIS2_EEE24__emplace_back_slow_pathIJS2_EEEvDpOT_(%"class.std::__1::vector"*, %"class.std::__1::future"*)

declare void @_ZNSt3__16futureIvED1Ev(%"class.std::__1::future"*)

declare void @_ZNSt3__117__assoc_sub_state4waitEv(%"class.std::__1::__assoc_sub_state"*)

declare void @_ZNKSt3__120__vector_base_commonILb1EE20__throw_length_errorEv(%union.anon.0*)

declare void @_ZNSt11logic_errorC2EPKc(%"class.std::logic_error"*, i8*)

declare void @_ZNSt12length_errorD1Ev(%"class.std::length_error"*)

declare void @__cxa_free_exception(i8*)

declare %union.ZHeader* @_ZNSt3__119__thread_local_dataEv()

declare i32 @pthread_setspecific(i64, i8*)

declare void @"_ZNSt3__119__async_assoc_stateIvNS_12__async_funcIZN5utilsL12parallel_forINS2_16execution_policyIvEEPxS6_Z32parallel_for_Integer64_Integer64E3$_0EE5errorRKT_T0_SC_T1_SD_OT2_EUlvE_JEEEED1Ev"(%"class.std::__1::__async_assoc_state"*)

declare void @"_ZNSt3__119__async_assoc_stateIvNS_12__async_funcIZN5utilsL12parallel_forINS2_16execution_policyIvEEPxS6_Z32parallel_for_Integer64_Integer64E3$_0EE5errorRKT_T0_SC_T1_SD_OT2_EUlvE_JEEEED0Ev"(%"class.std::__1::__async_assoc_state"*)

declare void @"_ZNSt3__119__async_assoc_stateIvNS_12__async_funcIZN5utilsL12parallel_forINS2_16execution_policyIvEEPxS6_Z32parallel_for_Integer64_Integer64E3$_0EE5errorRKT_T0_SC_T1_SD_OT2_EUlvE_JEEEE16__on_zero_sharedEv"(%"class.std::__1::__async_assoc_state"*)

declare void @"_ZNSt3__119__async_assoc_stateIvNS_12__async_funcIZN5utilsL12parallel_forINS2_16execution_policyIvEEPxS6_Z32parallel_for_Integer64_Integer64E3$_0EE5errorRKT_T0_SC_T1_SD_OT2_EUlvE_JEEEE9__executeEv"(%"class.std::__1::__async_assoc_state"*)

declare void @_ZNSt3__117__assoc_sub_state9set_valueEv(%"class.std::__1::__assoc_sub_state"*)

declare void @_ZSt17current_exceptionv(%struct.st_KMutex*)

declare void @_ZNSt3__117__assoc_sub_state13set_exceptionESt13exception_ptr(%"class.std::__1::__assoc_sub_state"*, %struct.st_KMutex*)

declare void @_ZNSt13exception_ptrD1Ev(%struct.st_KMutex*)

declare void @_ZNSt3__117__assoc_sub_state16__on_zero_sharedEv(%"class.std::__1::__assoc_sub_state"*)

declare void @_ZNSt3__118condition_variableD1Ev(%"class.std::__1::condition_variable"*)

declare void @_ZNSt3__15mutexD1Ev(%"class.std::__1::mutex"*)

declare void @_ZNSt3__114__shared_countD2Ev(%"class.std::__1::__shared_count"*)

declare void @_Z11InitHashingi(i32)

declare void @set_default_hash_function(i8*)

declare i32 @strcmp(i8*, i8*)

declare void @_Z14init_byte_hashi(i32)

declare i32 @crc_32_hash(i8*, i64)

declare i64 @crc_64_hash(i8*, i64)

declare void @_Z13init_crc_hashi(i32)

declare i32 @fnv_32_hash(i8*, i64)

declare i64 @fnv_64_hash(i8*, i64)

declare void @_Z13init_fnv_hashi(i32)

declare %struct.st_IntegerArrayHashTable* @New_IntegerArrayHashTable(i64, i64, i32, void (%struct.st_hash_entry*)*)

declare void @_ZL22IntegerArrayHashDeleteP13st_hash_entry(%struct.st_hash_entry*)

declare void @_ZL26IntegerArrayHashCopyDeleteP13st_hash_entry(%struct.st_hash_entry*)

declare i64 @_ZL27IntegerArrayPositionHashFunP13st_hash_entry(%struct.st_hash_entry*)

declare void @_ZL19IntegerArrayHashSetP13st_hash_entryS0_(%struct.st_hash_entry*, %struct.st_hash_entry*)

declare i32 @_ZL29IntegerArrayPositionHashEqualP13st_hash_entryS0_(%struct.st_hash_entry*, %struct.st_hash_entry*)

declare void @_ZL19IntegerArrayHashNewP13st_hash_entry(%struct.st_hash_entry*)

declare i64 @_ZL19IntegerArrayHashFunP13st_hash_entry(%struct.st_hash_entry*)

declare void @_ZL23IntegerArrayHashCopySetP13st_hash_entryS0_(%struct.st_hash_entry*, %struct.st_hash_entry*)

declare i32 @_ZL21IntegerArrayHashEqualP13st_hash_entryS0_(%struct.st_hash_entry*, %struct.st_hash_entry*)

declare void @Delete_IntegerArrayHashTable(%struct.st_IntegerArrayHashTable*)

declare %struct.st_hash_entry* @IntegerArrayHashTable_GetOrAdd(%struct.st_IntegerArrayHashTable*, i64*, i32*)

declare %struct.st_hash_entry* @IntegerArrayHashTable_Add(%struct.st_IntegerArrayHashTable*, i64*)

declare %struct.st_hash_entry* @IntegerArrayHashTable_Get(%struct.st_IntegerArrayHashTable*, i64*)

declare i32 @IntegerArrayHashTable_Remove(%struct.st_IntegerArrayHashTable*, %struct.st_hash_entry*)

declare i64 @IntegerArrayHashTable_Length(%struct.st_IntegerArrayHashTable*)

declare i64 @IntegerArrayHashTable_ByteCount(%struct.st_IntegerArrayHashTable*, i64 (%struct.st_hash_entry*)*)

declare void @_Z17init_hash_integeri(i32)

declare void @FreeString(%struct.st_hash_entry*)

declare void @HashEntry_NewGeneral(%struct.st_hash_entry*)

declare i64 @StringHash(%struct.st_hash_entry*)

declare void @SetHashString(%struct.st_hash_entry*, %struct.st_hash_entry*)

declare i32 @EqualHashString(%struct.st_hash_entry*, %struct.st_hash_entry*)

declare %struct.st_hashtable* @New_HashTable(i64, i64 (%struct.st_hash_entry*)*, void (%struct.st_hash_entry*, %struct.st_hash_entry*)*, i32 (%struct.st_hash_entry*, %struct.st_hash_entry*)*, void (%struct.st_hash_entry*)*, void (%struct.st_hash_entry*)*)

declare void @FreeOrEmpty_HashTable(%struct.st_hashtable*, i32)

declare i8* @HashTable_GetValue(%struct.st_hashtable*, i8*)

declare %struct.st_hash_entry* @HashTable_GetUsingFunction(%struct.st_hashtable*, i8*)

declare %struct.st_hash_entry* @HashTable_Add(%struct.st_hashtable*, i8*)

declare %struct.st_hash_entry* @HashTable_GetOrAdd(%struct.st_hashtable*, i8*, i32*)

declare i32 @HashTable_RemoveElem(%struct.st_hashtable*, i8*)

declare i32 @HashTable_RemoveCommon(%struct.st_hashtable*, %struct.st_hash_entry*, i32)

declare void @HashTable_AddValue(%struct.st_hashtable*, i8*, i8*)

declare i32 @HashTable_OrOverAllEntries(%struct.st_hashtable*, i32 (%struct.st_hash_entry*, i8*)*, i8*)

declare i64 @HashTable_ByteCount(%struct.st_hashtable*, i64 (%struct.st_hash_entry*)*)

declare void @_Z14init_hashtablei(i32)

declare i32 @murmur3_32_hash(i8*, i64)

declare i32 @murmur3_32_128_hash(i8*, i64)

declare i32 @murmur3_32_32_hash(i8*, i64)

declare i64 @murmur3_64_128_hash(i8*, i64)

declare void @_Z17init_murmur3_hashi(i32)

declare i32 @sip24_32_hash(i8*, i64)

declare i64 @sip24_64_hash(i8*, i64)

declare void @_Z15init_sip24_hashi(i32)

declare void @_Z16InitRTLConstantsi(i32)

declare void @StartOutOfMemoryAbort(i64)

declare i32 @watchhandle_c(i8*)

declare i32 @IntFromMint(i32*, i64)

declare i32 @IntFromMintArray(i32*, i64*, i64)

declare void @MintFromIntArray(i64*, i32*, i64)

declare i32 @MInt32Q(i64)

declare void @DeinitializeRuntimeLibrary(i32)

declare void @ReinitializeRuntimeLibrary()

declare void @InitRuntimeLibrary(i32)

declare void @InitializeRuntimeLibrary()

declare void @_Z15InitInterpreteri(i32)

declare void @AddInterpreterFunctionCall(i8*, i32 (%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)*, i32)

declare {}* @GetFunctionCallFunction(i8*)

declare i32 @_ZL18interpret_fc_errorP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @GetFunctionCallFlags(i8*)

declare void @InitializeFunctionCalls()

declare void @DeinitializeFunctionCalls()

declare void @_Z19init_function_callsi(i32)

declare i32 @_ZL23interpret_fc_dimensionsP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL17interpret_fc_sortP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL21interpret_fc_orderingP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL29interpret_fc_ordering_integerP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL26interpret_fc_integerdigitsP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL19interpret_fc_medianP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL17interpret_fc_takeP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL17interpret_fc_dropP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL24interpret_fc_min_integerP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL31interpret_fc_min_integer_tensorP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL32interpret_fc_min_integer_generalP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL21interpret_fc_min_realP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL28interpret_fc_min_real_tensorP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL29interpret_fc_min_real_generalP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL24interpret_fc_max_integerP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL31interpret_fc_max_integer_tensorP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL32interpret_fc_max_integer_generalP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL21interpret_fc_max_realP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL28interpret_fc_max_real_tensorP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL29interpret_fc_max_real_generalP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL19interpret_fc_dot_vvP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL16interpret_fc_dotP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL23interpret_fc_outer_listP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL24interpret_fc_outer_timesP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL23interpret_fc_outer_plusP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL19interpret_fc_insertP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL19interpret_fc_deleteP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL20interpret_fc_reverseP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL17interpret_fc_joinP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL26interpret_fc_join_at_levelP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL18interpret_fc_unionP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL23interpret_fc_complementP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL25interpret_fc_intersectionP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL24interpret_fc_rotate_leftP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL25interpret_fc_rotate_rightP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL35interpret_fc_iterator_count_integerP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL32interpret_fc_iterator_count_realP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL20interpret_fc_flattenP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL21interpret_fc_positionP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL18interpret_fc_countP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL20interpret_fc_memberQP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL18interpret_fc_freeQP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL21interpret_fc_orderedQP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL22interpret_fc_partitionP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL22interpret_fc_transposeP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL32interpret_fc_conjugate_transposeP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL21interpret_fc_pad_leftP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL22interpret_fc_pad_rightP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL18interpret_fc_totalP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL22interpret_fc_total_allP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL23interpret_fc_accumulateP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL24interpret_fc_vector_normP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL24interpret_fc_differencesP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL19interpret_fc_ratiosP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL22interpret_fc_small_detP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL26interpret_fc_RandomIntegerP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL27interpret_fc_RandomIntegersP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL23interpret_fc_RandomRealP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL24interpret_fc_RandomRealsP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL26interpret_fc_RandomComplexP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL28interpret_fc_RandomComplexesP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL25interpret_fc_RandomChoiceP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL32interpret_fc_RandomChoiceWeightsP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL25interpret_fc_RandomSampleP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL32interpret_fc_RandomSampleWeightsP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL23interpret_fc_SeedRandomP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL29interpret_fc_BlockRandomBeginP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL27interpret_fc_BlockRandomEndP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL25interpret_fc_RandomNormalP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL23interpret_fc_RandomBetaP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL25interpret_fc_LegacyRandomP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL18interpret_fc_tallyP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL16interpret_fc_bagP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL28interpret_fc_stuffbag_scalarP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL21interpret_fc_stuffbagP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL20interpret_fc_bagpartP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL27interpret_fc_compare_tensorP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL26interpret_fc_coerce_tensorP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL24interpret_fc_copy_tensorP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL18interpret_fc_crossP21st_WolframLibraryDataxP9MArgumentS1_(%struct.st_WolframLibraryData.208*, i64, %union.MArgument*, i32*)

declare i32 @_ZL20CompiledRandomSamplePP13st_MDataArrayS0_S0_x(%struct.st_MDataArray**, %struct.st_MDataArray*, %struct.st_MDataArray*, i64)

declare i32 @_ZL20CompiledRandomChoicePvP13st_MDataArrayS1_S1_(i8*, %struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*)

; Function Attrs: argmemonly nounwind
declare void @llvm.memmove.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #1

declare i32 @_ZL26interpret_fc_takedrop_implxP9MArgumentS_i(i64, %union.MArgument*, i32*, i32)

declare i32 @InsertTensorToCompiledTable(%struct.st_MDataArray*, %struct.st_MDataArray*, i64*)

declare i32 @CompiledGetTensorElement(%struct.st_MDataArray**, %struct.st_MDataArray*, i64, i64)

declare i32 @CompiledSetTensorElement(%struct.st_MDataArray*, %struct.st_MDataArray*, i64, i64)

declare i64 @CompiledPartIndexSize(i64, i32*, i8**, %struct.part_ind*)

declare void @CompiledPartToIndex(i8*, i64, %struct.part_ind*, i64**)

declare i32 @CompiledCollectedPart(%struct.st_MDataArray**, %struct.st_MDataArray*, i64, i64, i32*, i8**)

declare i32 @CompiledCollectedPartNumericArray(%struct.st_MDataArray**, %struct.st_MDataArray*, i64, i64, i32*, i8**)

declare i32 @CompiledCollectedSetPart(%struct.st_MDataArray*, %struct.st_MDataArray*, i64, i64, i32*, i8**)

declare i32 @GetStartEndPosForTakeOrDrop(i64, i64*, i64*, i64*, i64*)

declare i32 @CompareReals(i32, double, i64, double*)

declare i32 @CompareRealsNoTolerance(i32, i64, double*)

declare i32 @CompareComplexes(i32, double, i64, %"class.std::__1::complex"*, i32*)

declare i32 @CompareComplexesNoTolerance(i32, i64, %"class.std::__1::complex"*, i32*)

declare i32 @CompareMTensors(i32, double, i64, %struct.st_MDataArray**)

declare i32 @IteratorCountI(i64, i64, i64, i64*)

declare i32 @IteratorCountR(double, double, double, i64*)

declare %struct.st_SparseArrayAllocation* @SparseArrayAllocation_new()

declare void @SparseArrayAllocation_delete(%struct.st_SparseArrayAllocation**)

declare %struct.st_MTensorSparseArrayAllocation* @MTensorSparseArrayAllocation_new()

declare void @MTensorSparseArrayAllocation_delete(%struct.st_MTensorSparseArrayAllocation**)

declare i32 @ConsistentSparseArrayAllocationQ(%struct.st_SparseArrayAllocation*, %struct.st_SparseArrayAllocation*)

declare void @MSparseArray_deleteSparseArray(%struct.st_MSparseArray*)

declare %struct.st_SparseArraySkeleton* @SparseArraySkeleton_new(i64)

declare void @SparseArraySkeleton_delete(%struct.st_SparseArraySkeleton*)

declare %struct.st_MSparseArray* @MSparseArray_new(i64)

declare void @MSparseArray_delete(%struct.st_MSparseArray*)

declare void @SparseArraySkeleton_allocIndexMTensors(%struct.st_SparseArraySkeleton*, %struct.st_MDataArray**, %struct.st_MDataArray**)

declare void @MSparseArray_allocIndices(%struct.st_MSparseArray*)

declare %struct.st_MSparseArray* @MSparseArray_newAllocated(i64, i64*, i64)

declare %struct.st_MSparseArray* @MSparseArray_copy(%struct.st_MSparseArray*)

declare void @MSparseArray_setImplicitValue(%struct.st_MSparseArray*, i32, i8*)

declare %struct.st_MDataArray* @MSparseArray_allocValues(%struct.st_MSparseArray*, i32)

declare i64 @MSparseArray_byteCount(%struct.st_MSparseArray*)

declare void @MSparseArray_getData(%struct.st_MSparseArray*, i64*, i64**, i64*, i64**, i64**, i32*, %struct.st_MDataArray**, %struct.st_MDataArray**)

declare i32 @SparseArray_explicitPositions(%struct.st_SparseArraySkeleton*, %struct.st_MDataArray**)

declare i32 @MSparseArray_fromPositions(%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MSparseArray**)

declare i32 @MSparseArray_fromMTensor(%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MSparseArray**)

declare i32 @MSparseArray_resetImplicit(%struct.st_MSparseArray*, %struct.st_MDataArray*, %struct.st_MSparseArray**)

declare void @_Z16InitMSparseArrayi(i32)

declare %struct.MULTIPLE_ARRAY_HASH_STRUCT* @MultipleArrayHash_new(i64, i64*, i64, i64*, i64**)

declare void @MultipleArrayHash_clear(%struct.MULTIPLE_ARRAY_HASH_STRUCT*, i64)

declare void @MultipleArrayHash_delete(%struct.MULTIPLE_ARRAY_HASH_STRUCT*)

declare void @_Z15init_hash_indexi(i32)

declare i32 @MSparseArray_toMTensor(%struct.st_MSparseArray*, %struct.st_MDataArray**)

declare void @DSCTR(i64*, double*, i64*, double*)

declare void @ZSCTR(i64*, %"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*)

declare i32 @SparseVector_dot_SparseVector_mint(i64*, i64*, i64*, i64, i64*, i64*, i64, i64*, i64*, i32)

declare i32 @SparseVector_dot_SparseVector_double(double*, i64*, i64*, i64, i64*, double*, i64, i64*, double*, i32)

declare i32 @SparseVector_dot_SparseVector_complex(%"class.std::__1::complex"*, i64*, i64*, i64, i64*, %"class.std::__1::complex"*, i64, i64*, %"class.std::__1::complex"*, i32)

declare i32 @SparseVector_dot_mintVector(i64*, i64, i64*, i64*, i64*, i32)

declare i32 @SparseVector_dot_doubleVector(double*, i64, i64*, double*, double*, i32)

declare double @DDOTI(i64*, double*, i64*, double*)

declare i32 @SparseVector_dot_complexVector(%"class.std::__1::complex"*, i64, i64*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i32)

declare void @ZDOTUI(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*)

declare i32 @SparseMatrix_dot_SparseVector_mint(i64*, i64*, i64, i64, i64, i64*, i64*, i64*, i64, i64*, i64*, i64*, i32, i32)

declare i32 @SparseMatrix_dot_SparseVector_double(i64*, i64*, i64, i64, i64, i64*, i64*, double*, i64, i64*, double*, double*, i32, i32)

declare i32 @SparseMatrix_dot_SparseVector_complex(i64*, i64*, i64, i64, i64, i64*, i64*, %"class.std::__1::complex"*, i64, i64*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i32, i32)

declare i32 @SparseVector_dot_mintArray(i64*, i64, i64, i64*, i64*, i64*, i64, i32, i32)

declare i32 @SparseVector_dot_doubleArray(double*, i64, i64, i64*, double*, double*, i64, i32, i32)

declare i32 @SparseVector_dot_complexArray(%"class.std::__1::complex"*, i64, i64, i64*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64, i32, i32)

declare i64 @rowmaxnz(i64, i64*)

declare i32 @machfunc_icsrmv(i64, i64, i64*, i64*, i64*, i64*, i64*, i32, i32)

declare i32 @machfunc_dcsrmv(i64, i64, i64*, i64*, double*, double*, double*, i32, i32)

declare i32 @mkl_sparse_d_create_csr(%struct.sparse_matrix**, i32, i64, i64, i64*, i64*, i64*, double*)

declare i32 @mkl_sparse_d_mv(i32, double, %struct.sparse_matrix*, i64, i32, double*, double, double*)

declare i32 @mkl_sparse_destroy(%struct.sparse_matrix*)

declare i32 @machfunc_zcsrmv(i64, i64, i64*, i64*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i32, i32)

declare i32 @mkl_sparse_z_create_csr(%struct.sparse_matrix**, i32, i64, i64, i64*, i64*, i64*, %"class.std::__1::complex"*)

declare i32 @mkl_sparse_z_mv(i32, double, double, %struct.sparse_matrix*, i64, i32, %"class.std::__1::complex"*, double, double, %"class.std::__1::complex"*)

declare i32 @machfunc_icsrmm(i64, i64, i64, i64*, i64*, i64*, i64*, i64*, i32, i32)

declare void @_ZL11init_icsrmmPvP29st_ParallelThreadsEnvironment(i8*, %struct.st_ParallelThreadsEnvironment*)

declare void @_ZL10icsrmm_funPvP17st_ParallelThread(i8*, %struct.st_ParallelThread*)

declare i32 @machfunc_dcsrmm(i64, i64, i64, i64*, i64*, double*, double*, double*, i32, i32)

declare i32 @mkl_sparse_d_mm(i32, double, %struct.sparse_matrix*, i64, i32, i32, double*, i64, i64, double, double*, i64)

declare i32 @machfunc_zcsrmm(i64, i64, i64, i64*, i64*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i32, i32)

declare i32 @mkl_sparse_z_mm(i32, double, double, %struct.sparse_matrix*, i64, i32, i32, %"class.std::__1::complex"*, i64, i64, double, double, %"class.std::__1::complex"*, i64)

declare i32 @machfunc_icsrmultcsr(i64, i64, i64, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*, i32)

declare i32 @_ZL19csrmultcsr_symbolicxxxPxS_S_S_S_(i64, i64, i64*, i64*, i64*, i64*, i64*)

declare i32 @machfunc_dcsrmultcsr(i64, i64, i64, i64*, i64*, double*, i64*, i64*, double*, i64*, i64*, double*, i32)

declare i32 @machfunc_zcsrmultcsr(i64, i64, i64, i64*, i64*, %"class.std::__1::complex"*, i64*, i64*, %"class.std::__1::complex"*, i64*, i64*, %"class.std::__1::complex"*, i32)

declare i32 @machfunc_pcsrmultcsr(i64, i64, i64, i64*, i64*, i64*, i64*, i64*, i64*)

declare void @_Z21init_mach_sparse_blasi(i32)

declare i32 @SparseArraySkeleton_checkSorted(%struct.st_SparseArraySkeleton*)

declare void @_Z11InitMTensori(i32)

declare i64 @iabsmax(i64, i64*)

declare i32 @iscalarmul(i64*, i64, i64, i64, i32)

declare i32 @mblas_igemv(i8, i64, i64, i64*, i64*, i64*, i32)

declare i32 @dgemv_(i8*, i64*, i64*, double*, double*, i64*, double*, i64*, double*, double*, i64*)

declare void @_ZL10init_igemvPvP29st_ParallelThreadsEnvironment(i8*, %struct.st_ParallelThreadsEnvironment*)

declare void @_ZL9igemv_funPvP17st_ParallelThread(i8*, %struct.st_ParallelThread*)

declare i32 @TensorDot_igemv(%struct.st_MDataArray**, i8, %struct.st_MDataArray*, %struct.st_MDataArray*, i32)

declare i32 @mblas_igemm(i64, i64, i64, i64*, i64*, i64*, i32)

declare i32 @dgemm_(i8*, i8*, i64*, i64*, i64*, double*, double*, i64*, double*, i64*, double*, double*, i64*)

declare void @_ZL10init_igemmPvP29st_ParallelThreadsEnvironment(i8*, %struct.st_ParallelThreadsEnvironment*)

declare void @_ZL9igemm_funPvP17st_ParallelThread(i8*, %struct.st_ParallelThread*)

declare i32 @TensorDot_igemm(%struct.st_MDataArray**, %struct.st_MDataArray*, %struct.st_MDataArray*, i32)

declare i32 @IDCNumToType(%struct.idc_num_t*, i32)

declare i64 @MTensor_getByteCount(%struct.st_MDataArray*)

declare %struct.st_MDataArray* @MTensor_new()

declare %struct.st_MDataArray* @MTensor_newAllocatedCommon(i32, i64, i64*, i32)

declare i32 @MTensor_allocateMemoryCommon(%struct.st_MDataArray*, i32, i64, i64*, i32)

declare i32 @_ZL20MTensor_allocateDataP13st_MDataArrayixPKxxi(%struct.st_MDataArray*, i32, i64, i64, i32)

declare void @MTensor_delete(%struct.st_MDataArray*)

declare void @MTensor_freeMemory(%struct.st_MDataArray*)

declare void @MTensorArray_delete(i64, %struct.st_MDataArray**)

declare i64 @NumberOfElements_common(i64, i64*, i32)

declare %struct.st_MDataArray* @MTensor_newUsingDataArray(i32, i64, i64*, i8*)

declare %struct.st_MDataArray* @MTensor_newCopyFromDataArray(i32, i64, i64*, i8*)

declare void @MTensor_resetDimensions(%struct.st_MDataArray*, i64, i64*)

declare void @MTensor_reallocateAndCopyData(%struct.st_MDataArray*)

declare i32 @MTensor_allocateCommon(%struct.st_MDataArray**, i32, i64, i64*, i32)

declare i32 @MTensor_allocateMemoryLikeCommon(%struct.st_MDataArray*, %struct.st_MDataArray*, i32)

declare i32 @iMTensor_copyUniqueInExpr(%struct.st_MDataArray*, %struct.st_MDataArray*)

declare i32 @MTensor_copyUniqueCommon(%struct.st_MDataArray*, %struct.st_MDataArray*, i32)

declare i32 @MTensor_copy(%struct.st_MDataArray*, %struct.st_MDataArray*)

declare i32 @MTensor_makeCopy(%struct.st_MDataArray**, %struct.st_MDataArray*)

declare void @GenericData_copySomeCoerce(i8*, i32, i64, i64, i8*, i32, i64, i64, i64)

declare void @MTensor_copyContiguousData(%struct.st_MDataArray*, i64, %struct.st_MDataArray*, i64, i64)

declare void @MTensor_copySomeData(%struct.st_MDataArray*, i64, i64, %struct.st_MDataArray*, i64, i64, i64)

declare void @MTensor_copySomeDataCoerce(%struct.st_MDataArray*, i64, i64, %struct.st_MDataArray*, i64, i64, i64)

declare double @TensorDataToReal(%struct.st_MDataArray*, i64, i32*)

declare void @MTensor_initialize(%struct.st_MDataArray*, i64)

declare i32 @MTensor_maxCompatibleDimensions(%struct.st_MDataArray*, %struct.st_MDataArray*, i64*, i64**)

declare i32 @MTensor_compatibleDimensionsQ(%struct.st_MDataArray*, %struct.st_MDataArray*)

declare i32 @MTensor_equivalentQ(%struct.st_MDataArray*, %struct.st_MDataArray*)

declare %struct.st_MDataArray* @LoadNewRankZeroMTensor(i8*, i32)

declare %struct.st_MDataArray** @LoadSharedRankZeroMTensor(i8*, i32, i64)

declare void @InitializeMTensorThreads()

declare void @DeinitializeMTensorThreads()

declare void @_Z12init_mtensori(i32)

declare void @_GLOBAL__sub_I_m_tensor_memorytrack.cpp()

declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*)

declare void @TensorMemory_addNew(%struct.st_MDataArray*)

declare void @_ZNSt3__15mutex4lockEv(%"class.std::__1::mutex"*)

declare void @_ZNSt3__15mutex6unlockEv(%"class.std::__1::mutex"*)

declare void @TensorMemory_free(%struct.st_MDataArray*)

declare void @initializeTensorMemoryCheck(void (i8*)*)

declare i64 @_ZL6key_fnP13st_hash_entry(%struct.st_hash_entry*)

declare void @_ZL6set_fnP13st_hash_entryS0_(%struct.st_hash_entry*, %struct.st_hash_entry*)

declare i32 @_ZL8equal_fnP13st_hash_entryS0_(%struct.st_hash_entry*, %struct.st_hash_entry*)

declare void @_ZL6new_fnP13st_hash_entry(%struct.st_hash_entry*)

declare void @_ZL10destroy_fnP13st_hash_entry(%struct.st_hash_entry*)

declare void @runTensorMemoryCheck(i32)

declare i32 @Tensor_select_or_set_part(%struct.st_MDataArray*, %struct.st_MDataArray*, i64, %struct.part_ind*, i32)

declare void @_ZL28DataArray_select_or_set_partP13st_MDataArrayS0_xP8part_indi(%struct.st_MDataArray*, %struct.st_MDataArray*, i64, %struct.part_ind*, i32)

declare void @_ZL29CopySpecifiedPartsOfDataArrayP13st_MDataArrayxS0_xP8part_indi(%struct.st_MDataArray*, i64, %struct.st_MDataArray*, i64, %struct.part_ind*, i32)

declare void @_Z32tplCopySpecifiedPartsOfDataArrayIaEvP13st_MDataArrayxS1_xP8part_indi(%struct.st_MDataArray*, i64, %struct.st_MDataArray*, i64, %struct.part_ind*, i32)

declare void @_Z32tplCopySpecifiedPartsOfDataArrayIhEvP13st_MDataArrayxS1_xP8part_indi(%struct.st_MDataArray*, i64, %struct.st_MDataArray*, i64, %struct.part_ind*, i32)

declare void @_Z32tplCopySpecifiedPartsOfDataArrayIsEvP13st_MDataArrayxS1_xP8part_indi(%struct.st_MDataArray*, i64, %struct.st_MDataArray*, i64, %struct.part_ind*, i32)

declare void @_Z32tplCopySpecifiedPartsOfDataArrayItEvP13st_MDataArrayxS1_xP8part_indi(%struct.st_MDataArray*, i64, %struct.st_MDataArray*, i64, %struct.part_ind*, i32)

declare void @_Z32tplCopySpecifiedPartsOfDataArrayIiEvP13st_MDataArrayxS1_xP8part_indi(%struct.st_MDataArray*, i64, %struct.st_MDataArray*, i64, %struct.part_ind*, i32)

declare void @_Z32tplCopySpecifiedPartsOfDataArrayIjEvP13st_MDataArrayxS1_xP8part_indi(%struct.st_MDataArray*, i64, %struct.st_MDataArray*, i64, %struct.part_ind*, i32)

declare void @_Z32tplCopySpecifiedPartsOfDataArrayIxEvP13st_MDataArrayxS1_xP8part_indi(%struct.st_MDataArray*, i64, %struct.st_MDataArray*, i64, %struct.part_ind*, i32)

declare void @_Z32tplCopySpecifiedPartsOfDataArrayIyEvP13st_MDataArrayxS1_xP8part_indi(%struct.st_MDataArray*, i64, %struct.st_MDataArray*, i64, %struct.part_ind*, i32)

declare void @_Z32tplCopySpecifiedPartsOfDataArrayIfEvP13st_MDataArrayxS1_xP8part_indi(%struct.st_MDataArray*, i64, %struct.st_MDataArray*, i64, %struct.part_ind*, i32)

declare void @_Z32tplCopySpecifiedPartsOfDataArrayIdEvP13st_MDataArrayxS1_xP8part_indi(%struct.st_MDataArray*, i64, %struct.st_MDataArray*, i64, %struct.part_ind*, i32)

declare void @_Z32tplCopySpecifiedPartsOfDataArrayINSt3__17complexIfEEEvP13st_MDataArrayxS4_xP8part_indi(%struct.st_MDataArray*, i64, %struct.st_MDataArray*, i64, %struct.part_ind*, i32)

declare void @_Z32tplCopySpecifiedPartsOfDataArrayINSt3__17complexIdEEEvP13st_MDataArrayxS4_xP8part_indi(%struct.st_MDataArray*, i64, %struct.st_MDataArray*, i64, %struct.part_ind*, i32)

declare i32 @NumericArray_select_or_set_part(%struct.st_MDataArray*, %struct.st_MDataArray*, i64, %struct.part_ind*, i32)

declare i32 @NormalizeSeqSpec(i64, i32, i64*, i64*, i64*, i64*)

declare i32 @part_check_tensor_dims_impl(i64, i64*, i64, %struct.part_ind*, i64*, %struct.part_message_data_struct*)

declare i32 @setpart_check_tensor_impl(%struct.st_MDataArray*, %struct.st_MDataArray*, i64, %struct.part_ind*, %struct.part_message_data_struct*)

declare i64 @_ZL8LoopII_IP13st_MDataArrayS0_S0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i64 @_ZL7LoopOp2P13st_MDataArrayS0_S0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i64 @_ZL8LoopIR_IP13st_MDataArrayS0_S0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i64 @_ZL8LoopIR_RP13st_MDataArrayS0_S0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i64 @_ZL8LoopIR_CP13st_MDataArrayS0_S0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i64 @_ZL8LoopIC_RP13st_MDataArrayS0_S0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i64 @_ZL8LoopIC_CP13st_MDataArrayS0_S0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i64 @_ZL8LoopRI_IP13st_MDataArrayS0_S0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i64 @_ZL8LoopRI_RP13st_MDataArrayS0_S0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i64 @_ZL8LoopRI_CP13st_MDataArrayS0_S0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i64 @_ZL8LoopRR_IP13st_MDataArrayS0_S0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i64 @_ZL8LoopRR_RP13st_MDataArrayS0_S0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i64 @_ZL8LoopRR_CP13st_MDataArrayS0_S0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i64 @_ZL8LoopRC_RP13st_MDataArrayS0_S0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i64 @_ZL8LoopRC_CP13st_MDataArrayS0_S0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i64 @_ZL8LoopCI_RP13st_MDataArrayS0_S0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i64 @_ZL8LoopCI_CP13st_MDataArrayS0_S0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i64 @_ZL8LoopCR_RP13st_MDataArrayS0_S0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i64 @_ZL8LoopCR_CP13st_MDataArrayS0_S0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i64 @_ZL8LoopCC_RP13st_MDataArrayS0_S0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i64 @_ZL8LoopCC_CP13st_MDataArrayS0_S0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i64 @_ZL7LoopI_IP13st_MDataArrayS0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i64 @_ZL7LoopOp1P13st_MDataArrayS0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i64 @_ZL7LoopR_IP13st_MDataArrayS0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i64 @_ZL7LoopR_RP13st_MDataArrayS0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i64 @_ZL7LoopR_CP13st_MDataArrayS0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i64 @_ZL7LoopC_IP13st_MDataArrayS0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i64 @_ZL7LoopC_RP13st_MDataArrayS0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i64 @_ZL7LoopC_CP13st_MDataArrayS0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i32 @TensorAddTo(%struct.st_MDataArray*, %struct.st_MDataArray*, i32)

declare i64 @_ZL16TensorOp2InPlaceP13st_MDataArrayS0_ij(%struct.st_MDataArray*, %struct.st_MDataArray*, i32, i32)

declare i32 @TensorDivideBy(%struct.st_MDataArray*, %struct.st_MDataArray*, i32)

declare i32 @TensorSubtractFrom(%struct.st_MDataArray*, %struct.st_MDataArray*, i32)

declare i32 @TensorTimesBy(%struct.st_MDataArray*, %struct.st_MDataArray*, i32)

declare i32 @TensorPlus(%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, i32)

declare i32 @TensorDivide(%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, i32)

declare i32 @TensorSubtract(%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, i32)

declare i32 @TensorTimes(%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*, i32)

declare i32 @Math_T_T(i32, i32, %struct.st_MDataArray*, i32, %struct.st_MDataArray**)

declare i32 @Math_V_V(i32, i32, i32, i8*, i32, i8*)

declare i32 @Math_TT_T(i32, i32, %struct.st_MDataArray*, %struct.st_MDataArray*, i32, %struct.st_MDataArray**)

declare i32 @Math_VV_V(i32, i32, i32, i8*, i32, i8*, i32, i8*)

declare i32 @CanCoerceTensor(%struct.st_MDataArray*, i32, double)

declare i32 @CoerceTensor(%struct.st_MDataArray*, i32, %struct.st_MDataArray*)

declare i32 @CoercePartOfTensor(%struct.st_MDataArray*, i32, %struct.st_MDataArray*, i64, i64)

declare i32 @MTensor_listPart(%struct.st_MDataArray**, %struct.st_MDataArray*, %struct.st_MDataArray*)

declare void @_ZL24MTensor_internalListPartP13st_MDataArrayS0_S0_(%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*)

declare i32 @MTensor_doPartialConjugate(%struct.st_MDataArray*, i64, i64, i64, i64, i64)

declare i32 @MRealArrayOverflowQ(double*, i64)

declare i32 @MComplexArrayOverflowQ(%"class.std::__1::complex"*, i64)

declare i32 @TensorOverflowQ(%struct.st_MDataArray*)

declare i32 @MTensor_take(%struct.st_MDataArray**, %struct.st_MDataArray*, i64, i64*, i64*, i64*, i32*)

declare i32 @MTensor_drop(%struct.st_MDataArray**, %struct.st_MDataArray*, i64, i64*, i64*, i64*, i32*)

declare void @_ZL11rTensorDropP13st_MDataArrayxxxPxS0_S1_(%struct.st_MDataArray*, i64, i64, i64, i64*, %struct.st_MDataArray*, i64*)

declare i32 @MTensor_outerList(%struct.st_MDataArray**, i64, %struct.st_MDataArray**, i64*)

declare i32 @MTensor_outerPlus(%struct.st_MDataArray**, i64, %struct.st_MDataArray**, i64*, i32)

declare i32 @MTensor_outerTimes(%struct.st_MDataArray**, i64, %struct.st_MDataArray**, i64*, i32)

declare i32 @PreparePositions(%struct.st_MDataArray*, i64*, i64, i64)

declare i32 @GetInsertedDimension(%struct.st_MDataArray*, i64*)

declare i32 @MTensor_insert(%struct.st_MDataArray**, %struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*)

declare i32 @MTensor_Join(%struct.st_MDataArray**, %struct.st_MDataArray**, i64)

declare i32 @MTensor_joinDimension(%struct.st_MDataArray*, i64, %struct.st_MDataArray**, i64)

declare i32 @MTensor_Union(%struct.st_MDataArray**, %struct.st_MDataArray**, i64)

declare i32 @MTensor_computeIC(%struct.st_MDataArray**, %struct.st_MDataArray**, i64, i32)

declare i32 @MTensor_toplevelDelete(%struct.st_MDataArray**, %struct.st_MDataArray*, %struct.st_MDataArray*)

declare i32 @MTensor_reverse(%struct.st_MDataArray**, %struct.st_MDataArray*)

declare i32 @MTensor_rotate(%struct.st_MDataArray**, %struct.st_MDataArray*, i64, i64*)

declare void @_ZL10RotateDataPxS_xS_S_x(i64*, i64*, i64, i64*, i64*, i64)

declare i32 @MTensor_flatten(%struct.st_MDataArray**, %struct.st_MDataArray*, i64)

declare i32 @TensorOrderedQ(%struct.st_MDataArray*)

declare i32 @MTensor_position(%struct.st_MDataArray**, %struct.st_MDataArray*, %struct.st_MDataArray*, double)

declare i32 @_ZL21IntPositionMatchPartQP13st_MDataArrayS0_xxd(%struct.st_MDataArray*, %struct.st_MDataArray*, i64, i64, double)

declare i32 @_ZL22RealPositionMatchPartQP13st_MDataArrayS0_xxd(%struct.st_MDataArray*, %struct.st_MDataArray*, i64, i64, double)

declare i32 @_ZL25ComplexPositionMatchPartQP13st_MDataArrayS0_xxd(%struct.st_MDataArray*, %struct.st_MDataArray*, i64, i64, double)

declare i64 @TensorCount(%struct.st_MDataArray*, %struct.st_MDataArray*, double)

declare i32 @TensorMemberQ(%struct.st_MDataArray*, %struct.st_MDataArray*, double)

declare i32 @TensorFreeQ(%struct.st_MDataArray*, %struct.st_MDataArray*, double)

declare i32 @TensorPartition(%struct.st_MDataArray**, %struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*)

declare i64* @_ZL19PartitionIntTensor0PxS_S_xxS_S_xx(i64*, i64*, i64*, i64, i64, i64*, i64*, i64, i64)

declare double* @_ZL20PartitionRealTensor0PdPxS0_xxS0_S0_xx(double*, i64*, i64*, i64, i64, i64*, i64*, i64, i64)

declare %"class.std::__1::complex"* @_ZL18PartitionCxTensor0PNSt3__17complexIdEEPxS3_xxS3_S3_xx(%"class.std::__1::complex"*, i64*, i64*, i64, i64, i64*, i64*, i64, i64)

declare %"class.std::__1::complex"* @_ZL21PartitionCxTensorDeepPNSt3__17complexIdEExxxxPxS3_x(%"class.std::__1::complex"*, i64, i64, i64, i64, i64*, i64*, i64)

declare %"class.std::__1::complex"* @_ZL14MakeCxPartDeepPNSt3__17complexIdEExxxxPx(%"class.std::__1::complex"*, i64, i64, i64, i64, i64*)

declare double* @_ZL23PartitionRealTensorDeepPdxxxxPxS0_x(double*, i64, i64, i64, i64, i64*, i64*, i64)

declare double* @_ZL16MakeRealPartDeepPdxxxxPx(double*, i64, i64, i64, i64, i64*)

declare i64* @_ZL22PartitionIntTensorDeepPxxxxxS_S_x(i64*, i64, i64, i64, i64, i64*, i64*, i64)

declare i64* @_ZL15MakeIntPartDeepPxxxxxS_(i64*, i64, i64, i64, i64, i64*)

declare i32 @TransposeComplexBlock(i64, i64, i64, %"class.std::__1::complex"*)

declare i32 @TransposeComplex(i64, i64, %"class.std::__1::complex"*)

declare i32 @PermutationType(%struct.st_MDataArray*)

declare i32 @TransposeResultDimensions(%struct.st_MDataArray*, i64, i64*, %struct.st_MDataArray*)

declare i32 @MTensor_12TransposeConjugate(%struct.st_MDataArray**, %struct.st_MDataArray*, i32)

declare void @_ZL26MTensor_TransposeSubmatrixP13st_MDataArrayxS0_xxxx(%struct.st_MDataArray*, i64, %struct.st_MDataArray*, i64, i64, i64, i64)

declare i32 @MTensor_ndimTranspose(%struct.st_MDataArray**, %struct.st_MDataArray*, %struct.st_MDataArray*, i32)

declare void @InitializePosition(i64, i64*)

declare i64 @FlatIndex(i64, i64*, i64*)

declare i32 @IncrementPosition(i64, i64*, i64*)

declare i32 @IncrementPositionBy(i64, i64*, i64*, i64)

declare i32 @MTensor_pad(%struct.st_MDataArray**, %struct.st_MDataArray*, %struct.st_MDataArray*, i64*, i64*)

declare void @_ZL4tilexP13st_MDataArrayxPxS0_S1_S1_S1_(i64, %struct.st_MDataArray*, i64, i64*, %struct.st_MDataArray*, i64*, i64*, i64*)

declare void @_ZL7overlayxP13st_MDataArrayxPxS0_xS1_S1_(i64, %struct.st_MDataArray*, i64, i64*, %struct.st_MDataArray*, i64, i64*, i64*)

declare i32 @MTensor_accumulate(%struct.st_MDataArray**, %struct.st_MDataArray*, i32)

declare i32 @MTensor_differences(%struct.st_MDataArray**, %struct.st_MDataArray*, i64, i64*, i64*, i32, i32)

declare double @MReal_smallDet(i64, double**)

declare i32 @CheckIntegerTensor(%struct.st_MDataArray*, i32)

declare i32 @TensorSameQ(double, i32, %struct.st_MDataArray*, %struct.st_MDataArray*)

declare i32 @MTensor_tally(%struct.st_MDataArray**, %struct.st_MDataArray*)

declare i64 @TensorFirstOrLastInOrder(%struct.st_MDataArray*, i64)

declare i32 @MergeSortPermutation_mint(i64*, i64*, i64*, i64, i64)

declare i32 @MergeSortPermutation_double(double*, i64*, i64*, i64, i64)

declare i32 @MTensor_computeSortPermutation(%struct.st_MDataArray*, %struct.st_MDataArray*)

declare i32 @_ZL28MergeSortPermutation_complexPKNSt3__17complexIdEEPxS4_xx(%"class.std::__1::complex"*, i64*, i64*, i64, i64)

declare i32 @MTensor_Sort(%struct.st_MDataArray**, %struct.st_MDataArray*)

declare i32 @machfunc_idot(i64*, i64*, i64, i64*, i64, i64, i32)

declare i32 @machfunc_ddot(double*, double*, i64, double*, i64, i64, i32)

declare double @ddot_(i64*, double*, i64*, double*, i64*)

declare i32 @machfunc_cdot(%"class.std::__1::complex"*, i32, %"class.std::__1::complex"*, i64, %"class.std::__1::complex"*, i64, i64, i32)

declare void @zdotc_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*)

declare void @zdotu_(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*)

declare void @mblas_zdotc(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*)

declare void @mblas_zdotu(%"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*)

declare i32 @machfunc_igemv(i8, i64, i64, i64*, i64*, i64, i64*, i64, i64*, i64*, i64, i32)

declare i32 @machfunc_dgemv(i64, i64, double*, double*, i64, double*, i64, double*, double*, i64, i32)

declare i32 @machfunc_cgemv(i64, i64, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64, %"class.std::__1::complex"*, i64, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64, i32)

declare i32 @zgemv_(i8*, i64*, i64*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*)

declare i32 @machfunc_dgemtv(i64, i64, double*, double*, i64, double*, i64, double*, double*, i64, i32)

declare i32 @machfunc_cgemtv(i64, i64, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64, %"class.std::__1::complex"*, i64, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64, i32)

declare i32 @TensorDotMatrixVector(%struct.st_MDataArray**, %struct.st_MDataArray*, %struct.st_MDataArray*, i32)

declare i32 @TensorDotVectorMatrix(%struct.st_MDataArray**, %struct.st_MDataArray*, %struct.st_MDataArray*, i32)

declare i32 @TensorGEMV(i8, i64, i64, %struct.idc_num_t*, %struct.st_MDataArray*, i64, %struct.st_MDataArray*, i64, i64, %struct.idc_num_t*, %struct.st_MDataArray*, i64, i64, i32)

declare i32 @TensorGER(i64, i64, %struct.idc_num_t*, %struct.st_MDataArray*, i64, i64, %struct.st_MDataArray*, i64, i64, %struct.st_MDataArray*, i64, i64, i32)

declare i32 @dger_(i64*, i64*, double*, double*, i64*, double*, i64*, double*, i64*)

declare i32 @zgerc_(i64*, i64*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*)

declare i32 @zgeru_(i64*, i64*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*)

declare i32 @machfunc_igemm(i8, i8, i64, i64, i64, i64*, i64*, i64, i64*, i64, i64*, i64*, i64, i32)

declare i32 @MReal_smallMM(double*, double*, double*, i64, i64, i64)

declare i32 @MComplex_smallMM(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64, i64, i64)

declare i32 @MReal_smallMTM(double*, double*, double*, i64, i64, i64)

declare i32 @MComplex_smallMTM(%"class.std::__1::complex"*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64, i64, i64)

declare i32 @TensorDotMatrixMatrix(%struct.st_MDataArray**, %struct.st_MDataArray*, %struct.st_MDataArray*, i32)

declare i32 @_ZL21machfunc_dzgemm_smallxxxP13st_MDataArrayS0_S0_(i64, i64, i64, %struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*)

declare i32 @TensorGEMM(i8, i8, i64, i64, i64, %struct.idc_num_t*, %struct.st_MDataArray*, i64, %struct.st_MDataArray*, i64, %struct.idc_num_t*, %struct.st_MDataArray*, i64, i32)

declare i32 @zgemm_(i8*, i8*, i64*, i64*, i64*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, i64*, %"class.std::__1::complex"*, %"class.std::__1::complex"*, i64*)

declare i32 @MTensor_nDot(%struct.st_MDataArray**, %struct.st_MDataArray*, %struct.st_MDataArray*, i32)

declare i32 @TensorTrace(%struct.st_MDataArray*, i8*, i32)

declare %struct.st_TensorProperty* @TensorProperty_new(i64)

declare void @TensorProperty_delete(%struct.st_TensorProperty*)

declare void @TensorProperty_copyValues(%struct.st_TensorProperty*, %struct.st_TensorProperty*)

declare %struct.st_TensorProperty* @TensorProperty_copy(%struct.st_TensorProperty*)

declare void @_Z17InitMNumericArrayi(i32)

declare float @double_to_float(double, i32*, i32)

declare i32 @MNumericArrayConvertMethod_fromString(i8*)

declare void @_ZNKSt3__121__basic_string_commonILb1EE20__throw_length_errorEv(%union.anon.0*)

declare %"struct.std::__1::__hash_node_base"* @_ZNSt3__112__hash_tableINS_17__hash_value_typeINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEE21MNumericArrayMethodIdEENS_22__unordered_map_hasherIS7_S9_NS_4hashIS7_EELb1EEENS_21__unordered_map_equalIS7_S9_NS_8equal_toIS7_EELb1EEENS5_IS9_EEE4findIS7_EENS_15__hash_iteratorIPNS_11__hash_nodeIS9_PvEEEERKT_(%"class.std::__1::__hash_table"*, %"class.std::__1::basic_string"*)

declare i64 @_ZNSt3__121__murmur2_or_cityhashImLm64EEclEPKvm(%union.anon.0*, i8*, i64)

declare i8* @MNumericArrayConvertMethodString(i32)

declare i32 @MNumericArrayType_fromString(i8*)

declare %"struct.std::__1::__hash_node_base.22"* @_ZNSt3__112__hash_tableINS_17__hash_value_typeINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEE19MNumericArrayTypeIdEENS_22__unordered_map_hasherIS7_S9_NS_4hashIS7_EELb1EEENS_21__unordered_map_equalIS7_S9_NS_8equal_toIS7_EELb1EEENS5_IS9_EEE4findIS7_EENS_15__hash_iteratorIPNS_11__hash_nodeIS9_PvEEEERKT_(%"class.std::__1::__hash_table.26"*, %"class.std::__1::basic_string"*)

declare i8* @MNumericArrayTypeString(i32)

declare void @DeinitializeMNumericArray()

declare i64 @MNumericArray_getArrayBytes(%struct.st_MDataArray*)

declare i64 @MNumericArray_getByteCount(%struct.st_MDataArray*)

declare %struct.st_MDataArray* @MNumericArray_newAllocated(i32, i64)

declare %struct.st_MDataArray* @_ZL32MNumericArray_newAllocatedCommonixPKxPKvb(i32, i64, i64*, i8*, i1)

declare %struct.st_MDataArray* @MNumericArray_newAllocatedRank(i32, i64, i64*)

declare %struct.st_MDataArray* @MNumericArray_newAllocatedRankUsingDims(i32, i64, i64*)

declare %struct.st_MDataArray* @MNumericArray_newUsingDataArray(i32, i64, i64*, i8*)

declare %struct.st_MDataArray* @MNumericArray_newUsingDataArrayAndDims(i32, i64, i64*, i8*)

declare void @MNumericArray_delete(%struct.st_MDataArray*)

declare %struct.st_MDataArray* @MNumericArray_copy(%struct.st_MDataArray*)

declare void @MNumericArray_copyContiguousData(%struct.st_MDataArray*, i64, %struct.st_MDataArray*, i64, i64)

declare void @MNumericArray_copySomeData(%struct.st_MDataArray*, i64, i64, %struct.st_MDataArray*, i64, i64, i64)

declare void @MNumericArray_resetDimensions(%struct.st_MDataArray*, i64, i64*)

declare void @_Z18init_mnumericarrayi(i32)

declare void @_ZNSt3__112__hash_tableINS_17__hash_value_typeINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEE21MNumericArrayMethodIdEENS_22__unordered_map_hasherIS7_S9_NS_4hashIS7_EELb1EEENS_21__unordered_map_equalIS7_S9_NS_8equal_toIS7_EELb1EEENS5_IS9_EEE8__rehashEm(%"class.std::__1::__hash_table"*, i64)

declare void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEC1ERKS5_(%"class.std::__1::basic_string"*, %"class.std::__1::basic_string"*)

declare void @_ZNSt3__112__hash_tableINS_17__hash_value_typeINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEE21MNumericArrayMethodIdEENS_22__unordered_map_hasherIS7_S9_NS_4hashIS7_EELb1EEENS_21__unordered_map_equalIS7_S9_NS_8equal_toIS7_EELb1EEENS5_IS9_EEE6rehashEm(%"class.std::__1::__hash_table"*, i64)

declare void @_ZNSt3__112__hash_tableINS_17__hash_value_typeINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEE19MNumericArrayTypeIdEENS_22__unordered_map_hasherIS7_S9_NS_4hashIS7_EELb1EEENS_21__unordered_map_equalIS7_S9_NS_8equal_toIS7_EELb1EEENS5_IS9_EEE6rehashEm(%"class.std::__1::__hash_table.26"*, i64)

declare i64 @_ZNSt3__112__next_primeEm(i64)

declare void @_ZNSt3__112__hash_tableINS_17__hash_value_typeINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEE19MNumericArrayTypeIdEENS_22__unordered_map_hasherIS7_S9_NS_4hashIS7_EELb1EEENS_21__unordered_map_equalIS7_S9_NS_8equal_toIS7_EELb1EEENS5_IS9_EEE8__rehashEm(%"class.std::__1::__hash_table.26"*, i64)

declare i32 @_Z25MNumericArray_equivalentQP13st_MDataArrayS0_(%struct.st_MDataArray*, %struct.st_MDataArray*)

declare i32 @MNumericArray_sameQ(double, i32, %struct.st_MDataArray*, %struct.st_MDataArray*)

declare i32 @_ZL24MNumericArray_sameQ_implRKdP13st_MDataArrayS2_(double*, %struct.st_MDataArray*, %struct.st_MDataArray*)

declare i32 @isInRangeUnsigned(double, i64)

declare i32 @isInRangeSigned(double, i64, i64)

declare i32 @MNumericArray_convert_common(%struct.st_MDataArray**, %struct.st_MDataArray*, i32, %struct.st_MNumericArrayConvertData*)

declare i32 @_ZL26MNumericArray_convert_implPP13st_MDataArrayS0_RKxRKP27st_MNumericArrayConvertData(%struct.st_MDataArray**, i32, i8*, i64*, %struct.st_MNumericArrayConvertData*)

declare void @.omp_outlined..657(i32*, i32*, i64*, i64, i8**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..2.658(i32*, i32*, i64*, i64, i8**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..4.659(i32*, i32*, i64*, i64, i8**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..6.660(i32*, i32*, i64*, i64, i8**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..8.661(i32*, i32*, i64*, i64, i16**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..10.662(i32*, i32*, i64*, i64, i16**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..12.663(i32*, i32*, i64*, i64, i16**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..14.664(i32*, i32*, i64*, i64, i16**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..16.665(i32*, i32*, i64*, i64, i32**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..18.666(i32*, i32*, i64*, i64, i32**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..20.667(i32*, i32*, i64*, i64, i32**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..22.668(i32*, i32*, i64*, i64, i32**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..24.669(i32*, i32*, i64*, i64, i64**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..26.670(i32*, i32*, i64*, i64, i64**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..28.671(i32*, i32*, i64*, i64, i64**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..30.672(i32*, i32*, i64*, i64, i64**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..32.673(i32*, i32*, i64*, i64, i8**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..34.674(i32*, i32*, i64*, i64, i8**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..36.675(i32*, i32*, i64*, i64, i8**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..38(i32*, i32*, i64*, i64, i8**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..40(i32*, i32*, i64*, i64, i16**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..42(i32*, i32*, i64*, i64, i16**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..44(i32*, i32*, i64*, i64, i16**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..46(i32*, i32*, i64*, i64, i16**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..48(i32*, i32*, i64*, i64, i32**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..50(i32*, i32*, i64*, i64, i32**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..52.676(i32*, i32*, i64*, i64, i32**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..54(i32*, i32*, i64*, i64, i32**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..56(i32*, i32*, i64*, i64, i64**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..58(i32*, i32*, i64*, i64, i64**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..60(i32*, i32*, i64*, i64, i64**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..62(i32*, i32*, i64*, i64, i64**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..64(i32*, i32*, i64*, i64, float**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..66(i32*, i32*, i64*, i64, float**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..68(i32*, i32*, i64*, i64, float**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..70(i32*, i32*, i64*, i64, float**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..72(i32*, i32*, i64*, i64, double**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..74(i32*, i32*, i64*, i64, double**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..76(i32*, i32*, i64*, i64, double**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..78(i32*, i32*, i64*, i64, double**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..80(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..82(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..84(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..86(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..88(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..90(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..92(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..94(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..96(i32*, i32*, i64*, i64, i8**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..98(i32*, i32*, i64*, i64, i8**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..100(i32*, i32*, i64*, i64, i8**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..102(i32*, i32*, i64*, i64, i8**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..104(i32*, i32*, i64*, i64, i16**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..106(i32*, i32*, i64*, i64, i16**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..108(i32*, i32*, i64*, i64, i16**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..110(i32*, i32*, i64*, i64, i16**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..112(i32*, i32*, i64*, i64, i32**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..114(i32*, i32*, i64*, i64, i32**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..116(i32*, i32*, i64*, i64, i32**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..118(i32*, i32*, i64*, i64, i32**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..120(i32*, i32*, i64*, i64, i64**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..122(i32*, i32*, i64*, i64, i64**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..124(i32*, i32*, i64*, i64, i64**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..126(i32*, i32*, i64*, i64, i64**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..128(i32*, i32*, i64*, i64, i8**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..130(i32*, i32*, i64*, i64, i8**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..132(i32*, i32*, i64*, i64, i8**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..134(i32*, i32*, i64*, i64, i8**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..136(i32*, i32*, i64*, i64, i16**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..138(i32*, i32*, i64*, i64, i16**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..140(i32*, i32*, i64*, i64, i16**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..142(i32*, i32*, i64*, i64, i16**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..144(i32*, i32*, i64*, i64, i32**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..146(i32*, i32*, i64*, i64, i32**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..148(i32*, i32*, i64*, i64, i32**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..150(i32*, i32*, i64*, i64, i32**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..152(i32*, i32*, i64*, i64, i64**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..154(i32*, i32*, i64*, i64, i64**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..156(i32*, i32*, i64*, i64, i64**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..158(i32*, i32*, i64*, i64, i64**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..160(i32*, i32*, i64*, i64, float**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..162(i32*, i32*, i64*, i64, float**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..164(i32*, i32*, i64*, i64, float**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..166(i32*, i32*, i64*, i64, float**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..168(i32*, i32*, i64*, i64, double**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..170(i32*, i32*, i64*, i64, double**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..172(i32*, i32*, i64*, i64, double**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..174(i32*, i32*, i64*, i64, double**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..176(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..178(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..180(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..182(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..184(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..186(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..188(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..190(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..192(i32*, i32*, i64*, i64, i8**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..194(i32*, i32*, i64*, i64, i8**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..196(i32*, i32*, i64*, i64, i8**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..198(i32*, i32*, i64*, i64, i8**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..200(i32*, i32*, i64*, i64, i16**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..202(i32*, i32*, i64*, i64, i16**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..204(i32*, i32*, i64*, i64, i16**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..206(i32*, i32*, i64*, i64, i16**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..208(i32*, i32*, i64*, i64, i32**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..210(i32*, i32*, i64*, i64, i32**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..212(i32*, i32*, i64*, i64, i32**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..214(i32*, i32*, i64*, i64, i32**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..216(i32*, i32*, i64*, i64, i64**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..218(i32*, i32*, i64*, i64, i64**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..220(i32*, i32*, i64*, i64, i64**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..222(i32*, i32*, i64*, i64, i64**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..224(i32*, i32*, i64*, i64, i8**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..226(i32*, i32*, i64*, i64, i8**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..228(i32*, i32*, i64*, i64, i8**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..230(i32*, i32*, i64*, i64, i8**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..232(i32*, i32*, i64*, i64, i16**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..234(i32*, i32*, i64*, i64, i16**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..236(i32*, i32*, i64*, i64, i16**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..238(i32*, i32*, i64*, i64, i16**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..240(i32*, i32*, i64*, i64, i32**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..242(i32*, i32*, i64*, i64, i32**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..244(i32*, i32*, i64*, i64, i32**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..246(i32*, i32*, i64*, i64, i32**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..248(i32*, i32*, i64*, i64, i64**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..250(i32*, i32*, i64*, i64, i64**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..252(i32*, i32*, i64*, i64, i64**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..254(i32*, i32*, i64*, i64, i64**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..256(i32*, i32*, i64*, i64, float**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..258(i32*, i32*, i64*, i64, float**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..260(i32*, i32*, i64*, i64, float**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..262(i32*, i32*, i64*, i64, float**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..264(i32*, i32*, i64*, i64, double**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..266(i32*, i32*, i64*, i64, double**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..268(i32*, i32*, i64*, i64, double**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..270(i32*, i32*, i64*, i64, double**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..272(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..274(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..276(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..278(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..280(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..282(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..284(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..286(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..288(i32*, i32*, i64*, i64, i8**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..290(i32*, i32*, i64*, i64, i8**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..292(i32*, i32*, i64*, i64, i8**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..294(i32*, i32*, i64*, i64, i8**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..296(i32*, i32*, i64*, i64, i16**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..298(i32*, i32*, i64*, i64, i16**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..300(i32*, i32*, i64*, i64, i16**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..302(i32*, i32*, i64*, i64, i16**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..304(i32*, i32*, i64*, i64, i32**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..306(i32*, i32*, i64*, i64, i32**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..308(i32*, i32*, i64*, i64, i32**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..310(i32*, i32*, i64*, i64, i32**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..312(i32*, i32*, i64*, i64, i64**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..314(i32*, i32*, i64*, i64, i64**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..316(i32*, i32*, i64*, i64, i64**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..318(i32*, i32*, i64*, i64, i64**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..320(i32*, i32*, i64*, i64, i8**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..322(i32*, i32*, i64*, i64, i8**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..324(i32*, i32*, i64*, i64, i8**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..326(i32*, i32*, i64*, i64, i8**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..328(i32*, i32*, i64*, i64, i16**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..330(i32*, i32*, i64*, i64, i16**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..332(i32*, i32*, i64*, i64, i16**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..334(i32*, i32*, i64*, i64, i16**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..336(i32*, i32*, i64*, i64, i32**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..338(i32*, i32*, i64*, i64, i32**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..340(i32*, i32*, i64*, i64, i32**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..342(i32*, i32*, i64*, i64, i32**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..344(i32*, i32*, i64*, i64, i64**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..346(i32*, i32*, i64*, i64, i64**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..348(i32*, i32*, i64*, i64, i64**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..350(i32*, i32*, i64*, i64, i64**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..352(i32*, i32*, i64*, i64, float**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..354(i32*, i32*, i64*, i64, float**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..356(i32*, i32*, i64*, i64, float**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..358(i32*, i32*, i64*, i64, float**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..360(i32*, i32*, i64*, i64, double**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..362(i32*, i32*, i64*, i64, double**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..364(i32*, i32*, i64*, i64, double**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..366(i32*, i32*, i64*, i64, double**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..368(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..370(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..372(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..374(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..376(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..378(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..380(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..382(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..384(i32*, i32*, i64*, i64, i8**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..386(i32*, i32*, i64*, i64, i8**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..388(i32*, i32*, i64*, i64, i8**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..390(i32*, i32*, i64*, i64, i8**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..392(i32*, i32*, i64*, i64, i16**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..394(i32*, i32*, i64*, i64, i16**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..396(i32*, i32*, i64*, i64, i16**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..398(i32*, i32*, i64*, i64, i16**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..400(i32*, i32*, i64*, i64, i32**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..402(i32*, i32*, i64*, i64, i32**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..404(i32*, i32*, i64*, i64, i32**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..406(i32*, i32*, i64*, i64, i32**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..408(i32*, i32*, i64*, i64, i64**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..410(i32*, i32*, i64*, i64, i64**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..412(i32*, i32*, i64*, i64, i64**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..414(i32*, i32*, i64*, i64, i64**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..416(i32*, i32*, i64*, i64, i8**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..418(i32*, i32*, i64*, i64, i8**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..420(i32*, i32*, i64*, i64, i8**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..422(i32*, i32*, i64*, i64, i8**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..424(i32*, i32*, i64*, i64, i16**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..426(i32*, i32*, i64*, i64, i16**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..428(i32*, i32*, i64*, i64, i16**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..430(i32*, i32*, i64*, i64, i16**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..432(i32*, i32*, i64*, i64, i32**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..434(i32*, i32*, i64*, i64, i32**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..436(i32*, i32*, i64*, i64, i32**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..438(i32*, i32*, i64*, i64, i32**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..440(i32*, i32*, i64*, i64, i64**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..442(i32*, i32*, i64*, i64, i64**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..444(i32*, i32*, i64*, i64, i64**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..446(i32*, i32*, i64*, i64, i64**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..448(i32*, i32*, i64*, i64, float**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..450(i32*, i32*, i64*, i64, float**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..452(i32*, i32*, i64*, i64, float**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..454(i32*, i32*, i64*, i64, float**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..456(i32*, i32*, i64*, i64, double**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..458(i32*, i32*, i64*, i64, double**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..460(i32*, i32*, i64*, i64, double**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..462(i32*, i32*, i64*, i64, double**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..464(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..466(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..468(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..470(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..472(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..474(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..476(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..478(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i8**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..480(i32*, i32*, i64*, i64, i8**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..482(i32*, i32*, i64*, i64, i8**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..484(i32*, i32*, i64*, i64, i8**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..486(i32*, i32*, i64*, i64, i8**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..488(i32*, i32*, i64*, i64, i16**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..490(i32*, i32*, i64*, i64, i16**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..492(i32*, i32*, i64*, i64, i16**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..494(i32*, i32*, i64*, i64, i16**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..496(i32*, i32*, i64*, i64, i32**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..498(i32*, i32*, i64*, i64, i32**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..500(i32*, i32*, i64*, i64, i32**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..502(i32*, i32*, i64*, i64, i32**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..504(i32*, i32*, i64*, i64, i64**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..506(i32*, i32*, i64*, i64, i64**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..508(i32*, i32*, i64*, i64, i64**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..510(i32*, i32*, i64*, i64, i64**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..512(i32*, i32*, i64*, i64, i8**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..514(i32*, i32*, i64*, i64, i8**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..516(i32*, i32*, i64*, i64, i8**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..518(i32*, i32*, i64*, i64, i8**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..520(i32*, i32*, i64*, i64, i16**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..522(i32*, i32*, i64*, i64, i16**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..524(i32*, i32*, i64*, i64, i16**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..526(i32*, i32*, i64*, i64, i16**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..528(i32*, i32*, i64*, i64, i32**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..530(i32*, i32*, i64*, i64, i32**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..532(i32*, i32*, i64*, i64, i32**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..534(i32*, i32*, i64*, i64, i32**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..536(i32*, i32*, i64*, i64, i64**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..538(i32*, i32*, i64*, i64, i64**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..540(i32*, i32*, i64*, i64, i64**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..542(i32*, i32*, i64*, i64, i64**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..544(i32*, i32*, i64*, i64, float**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..546(i32*, i32*, i64*, i64, float**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..548(i32*, i32*, i64*, i64, float**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..550(i32*, i32*, i64*, i64, float**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..552(i32*, i32*, i64*, i64, double**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..554(i32*, i32*, i64*, i64, double**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..556(i32*, i32*, i64*, i64, double**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..558(i32*, i32*, i64*, i64, double**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..560(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..562(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..564(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..566(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..568(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..570(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..572(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..574(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i16**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..576(i32*, i32*, i64*, i64, i8**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..578(i32*, i32*, i64*, i64, i8**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..580(i32*, i32*, i64*, i64, i8**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..582(i32*, i32*, i64*, i64, i8**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..584(i32*, i32*, i64*, i64, i16**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..586(i32*, i32*, i64*, i64, i16**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..588(i32*, i32*, i64*, i64, i16**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..590(i32*, i32*, i64*, i64, i16**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..592(i32*, i32*, i64*, i64, i32**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..594(i32*, i32*, i64*, i64, i32**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..596(i32*, i32*, i64*, i64, i32**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..598(i32*, i32*, i64*, i64, i32**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..600(i32*, i32*, i64*, i64, i64**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..602(i32*, i32*, i64*, i64, i64**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..604(i32*, i32*, i64*, i64, i64**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..606(i32*, i32*, i64*, i64, i64**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..608(i32*, i32*, i64*, i64, i8**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..610(i32*, i32*, i64*, i64, i8**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..612(i32*, i32*, i64*, i64, i8**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..614(i32*, i32*, i64*, i64, i8**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..616(i32*, i32*, i64*, i64, i16**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..618(i32*, i32*, i64*, i64, i16**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..620(i32*, i32*, i64*, i64, i16**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..622(i32*, i32*, i64*, i64, i16**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..624(i32*, i32*, i64*, i64, i32**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..626(i32*, i32*, i64*, i64, i32**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..628(i32*, i32*, i64*, i64, i32**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..630(i32*, i32*, i64*, i64, i32**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..632(i32*, i32*, i64*, i64, i64**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..634(i32*, i32*, i64*, i64, i64**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..636(i32*, i32*, i64*, i64, i64**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..638(i32*, i32*, i64*, i64, i64**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..640(i32*, i32*, i64*, i64, float**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..642(i32*, i32*, i64*, i64, float**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..644(i32*, i32*, i64*, i64, float**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..646(i32*, i32*, i64*, i64, float**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..648(i32*, i32*, i64*, i64, double**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..650(i32*, i32*, i64*, i64, double**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..652(i32*, i32*, i64*, i64, double**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..654(i32*, i32*, i64*, i64, double**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..656(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..658(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..660(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..662(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..664(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..666(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..668(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..670(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i32**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..672(i32*, i32*, i64*, i64, i8**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..674(i32*, i32*, i64*, i64, i8**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..676(i32*, i32*, i64*, i64, i8**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..678(i32*, i32*, i64*, i64, i8**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..680(i32*, i32*, i64*, i64, i16**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..682(i32*, i32*, i64*, i64, i16**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..684(i32*, i32*, i64*, i64, i16**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..686(i32*, i32*, i64*, i64, i16**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..688(i32*, i32*, i64*, i64, i32**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..690(i32*, i32*, i64*, i64, i32**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..692(i32*, i32*, i64*, i64, i32**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..694(i32*, i32*, i64*, i64, i32**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..696(i32*, i32*, i64*, i64, i64**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..698(i32*, i32*, i64*, i64, i64**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..700(i32*, i32*, i64*, i64, i64**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..702(i32*, i32*, i64*, i64, i64**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..704(i32*, i32*, i64*, i64, i8**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..706(i32*, i32*, i64*, i64, i8**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..708(i32*, i32*, i64*, i64, i8**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..710(i32*, i32*, i64*, i64, i8**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..712(i32*, i32*, i64*, i64, i16**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..714(i32*, i32*, i64*, i64, i16**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..716(i32*, i32*, i64*, i64, i16**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..718(i32*, i32*, i64*, i64, i16**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..720(i32*, i32*, i64*, i64, i32**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..722(i32*, i32*, i64*, i64, i32**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..724(i32*, i32*, i64*, i64, i32**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..726(i32*, i32*, i64*, i64, i32**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..728(i32*, i32*, i64*, i64, i64**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..730(i32*, i32*, i64*, i64, i64**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..732(i32*, i32*, i64*, i64, i64**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..734(i32*, i32*, i64*, i64, i64**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..736(i32*, i32*, i64*, i64, float**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..738(i32*, i32*, i64*, i64, float**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..740(i32*, i32*, i64*, i64, float**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..742(i32*, i32*, i64*, i64, float**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..744(i32*, i32*, i64*, i64, double**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..746(i32*, i32*, i64*, i64, double**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..748(i32*, i32*, i64*, i64, double**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..750(i32*, i32*, i64*, i64, double**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..752(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..754(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..756(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..758(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..760(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..762(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..764(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..766(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, i64**, %struct.st_MNumericArrayConvertData**, i32*)

declare i32 @_ZL26MNumericArray_convert_implIfEiPP13st_MDataArrayPKT_RKxRKP27st_MNumericArrayConvertData(i32, i8*, float*, i64*, %struct.st_MNumericArrayConvertData*)

declare i32 @_ZL26MNumericArray_convert_implIdEiPP13st_MDataArrayPKT_RKxRKP27st_MNumericArrayConvertData(i32, i8*, double*, i64*, %struct.st_MNumericArrayConvertData*)

declare i32 @_ZL26MNumericArray_convert_implINSt3__17complexIfEEEiPP13st_MDataArrayPKT_RKxRKP27st_MNumericArrayConvertData(i32, i8*, %"class.std::__1::complex.156"*, i64*, %struct.st_MNumericArrayConvertData*)

declare i32 @_ZL26MNumericArray_convert_implINSt3__17complexIdEEEiPP13st_MDataArrayPKT_RKxRKP27st_MNumericArrayConvertData(i32, i8*, %"class.std::__1::complex"*, i64*, %struct.st_MNumericArrayConvertData*)

declare void @.omp_outlined..1056(i32*, i32*, i64*, i64, i8**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1058(i32*, i32*, i64*, i64, i8**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1060(i32*, i32*, i64*, i64, i8**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1062(i32*, i32*, i64*, i64, i8**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1064(i32*, i32*, i64*, i64, i16**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1066(i32*, i32*, i64*, i64, i16**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1068(i32*, i32*, i64*, i64, i16**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1070(i32*, i32*, i64*, i64, i16**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1072(i32*, i32*, i64*, i64, i32**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1074(i32*, i32*, i64*, i64, i32**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1076(i32*, i32*, i64*, i64, i32**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1078(i32*, i32*, i64*, i64, i32**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1080(i32*, i32*, i64*, i64, i64**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1082(i32*, i32*, i64*, i64, i64**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1084(i32*, i32*, i64*, i64, i64**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1086(i32*, i32*, i64*, i64, i64**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1088(i32*, i32*, i64*, i64, i8**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1090(i32*, i32*, i64*, i64, i8**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1092(i32*, i32*, i64*, i64, i8**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1094(i32*, i32*, i64*, i64, i8**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1096(i32*, i32*, i64*, i64, i16**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1098(i32*, i32*, i64*, i64, i16**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1100(i32*, i32*, i64*, i64, i16**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1102(i32*, i32*, i64*, i64, i16**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1104(i32*, i32*, i64*, i64, i32**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1106(i32*, i32*, i64*, i64, i32**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1108(i32*, i32*, i64*, i64, i32**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1110(i32*, i32*, i64*, i64, i32**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1112(i32*, i32*, i64*, i64, i64**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1114(i32*, i32*, i64*, i64, i64**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1116(i32*, i32*, i64*, i64, i64**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1118(i32*, i32*, i64*, i64, i64**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1120(i32*, i32*, i64*, i64, float**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare i32 @_ZN3wrt13mnumericarray7convertL10check_implIfdEENSt3__19enable_ifIXaaaasr17is_floating_pointIT0_EE5valuesr17is_floating_pointIT_EE5valuesr19is_larger_type_sizeIS5_S6_EE5valueEiE4typeEPS6_RKS5_P27st_MNumericArrayConvertData(float*, double*, %struct.st_MNumericArrayConvertData*)

declare void @.omp_outlined..1122(i32*, i32*, i64*, i64, float**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1124(i32*, i32*, i64*, i64, float**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1126(i32*, i32*, i64*, i64, float**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1128(i32*, i32*, i64*, i64, double**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1130(i32*, i32*, i64*, i64, double**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1132(i32*, i32*, i64*, i64, double**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1134(i32*, i32*, i64*, i64, double**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1136(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare i32 @_ZN3wrt13mnumericarray7convertL11coerce_implINSt3__17complexIfEENS4_IdEEEENS3_9enable_ifIXaasr10is_complexIT0_EE5valuesr10is_complexIT_EE5valueEiE4typeEPS9_RKS8_P27st_MNumericArrayConvertData(%"class.std::__1::complex.156"*, %"class.std::__1::complex"*, %struct.st_MNumericArrayConvertData*)

declare void @.omp_outlined..1138(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare i32 @_ZN3wrt13mnumericarray7convertL10round_implINSt3__17complexIfEENS4_IdEEEENS3_9enable_ifIXaasr10is_complexIT0_EE5valuesr10is_complexIT_EE5valueEiE4typeEPS9_RKS8_P27st_MNumericArrayConvertData(%"class.std::__1::complex.156"*, %"class.std::__1::complex"*, %struct.st_MNumericArrayConvertData*)

declare void @.omp_outlined..1140(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare i32 @_ZN3wrt13mnumericarray7convertL10scale_implINSt3__17complexIfEENS4_IdEEEENS3_9enable_ifIXaasr10is_complexIT0_EE5valuesr10is_complexIT_EE5valueEiE4typeEPS9_RKS8_P27st_MNumericArrayConvertData(%"class.std::__1::complex.156"*, %"class.std::__1::complex"*, %struct.st_MNumericArrayConvertData*)

declare void @.omp_outlined..1142(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1144(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1146(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1148(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1150(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, %"class.std::__1::complex"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..960(i32*, i32*, i64*, i64, i8**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..962(i32*, i32*, i64*, i64, i8**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..964(i32*, i32*, i64*, i64, i8**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..966(i32*, i32*, i64*, i64, i8**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..968(i32*, i32*, i64*, i64, i16**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..970(i32*, i32*, i64*, i64, i16**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..972(i32*, i32*, i64*, i64, i16**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..974(i32*, i32*, i64*, i64, i16**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..976(i32*, i32*, i64*, i64, i32**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..978(i32*, i32*, i64*, i64, i32**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..980(i32*, i32*, i64*, i64, i32**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..982(i32*, i32*, i64*, i64, i32**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..984(i32*, i32*, i64*, i64, i64**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..986(i32*, i32*, i64*, i64, i64**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..988(i32*, i32*, i64*, i64, i64**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..990(i32*, i32*, i64*, i64, i64**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..992(i32*, i32*, i64*, i64, i8**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..994(i32*, i32*, i64*, i64, i8**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..996(i32*, i32*, i64*, i64, i8**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..998(i32*, i32*, i64*, i64, i8**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1000(i32*, i32*, i64*, i64, i16**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1002(i32*, i32*, i64*, i64, i16**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1004(i32*, i32*, i64*, i64, i16**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1006(i32*, i32*, i64*, i64, i16**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1008(i32*, i32*, i64*, i64, i32**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1010(i32*, i32*, i64*, i64, i32**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1012(i32*, i32*, i64*, i64, i32**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1014(i32*, i32*, i64*, i64, i32**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1016(i32*, i32*, i64*, i64, i64**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1018(i32*, i32*, i64*, i64, i64**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1020(i32*, i32*, i64*, i64, i64**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1022(i32*, i32*, i64*, i64, i64**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1024(i32*, i32*, i64*, i64, float**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1026(i32*, i32*, i64*, i64, float**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1028(i32*, i32*, i64*, i64, float**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1030(i32*, i32*, i64*, i64, float**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1032(i32*, i32*, i64*, i64, double**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1034(i32*, i32*, i64*, i64, double**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1036(i32*, i32*, i64*, i64, double**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1038(i32*, i32*, i64*, i64, double**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1040(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1042(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1044(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1046(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1048(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1050(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..1052(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

; Function Attrs: nounwind readnone speculatable
declare <2 x float> @llvm.nearbyint.v2f32(<2 x float>) #0

declare void @.omp_outlined..1054(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, %"class.std::__1::complex.156"**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..864(i32*, i32*, i64*, i64, i8**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..866(i32*, i32*, i64*, i64, i8**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..868(i32*, i32*, i64*, i64, i8**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..870(i32*, i32*, i64*, i64, i8**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..872(i32*, i32*, i64*, i64, i16**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..874(i32*, i32*, i64*, i64, i16**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..876(i32*, i32*, i64*, i64, i16**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..878(i32*, i32*, i64*, i64, i16**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..880(i32*, i32*, i64*, i64, i32**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..882(i32*, i32*, i64*, i64, i32**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..884(i32*, i32*, i64*, i64, i32**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..886(i32*, i32*, i64*, i64, i32**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..888(i32*, i32*, i64*, i64, i64**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..890(i32*, i32*, i64*, i64, i64**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..892(i32*, i32*, i64*, i64, i64**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..894(i32*, i32*, i64*, i64, i64**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..896(i32*, i32*, i64*, i64, i8**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..898(i32*, i32*, i64*, i64, i8**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..900(i32*, i32*, i64*, i64, i8**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..902(i32*, i32*, i64*, i64, i8**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..904(i32*, i32*, i64*, i64, i16**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..906(i32*, i32*, i64*, i64, i16**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..908(i32*, i32*, i64*, i64, i16**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..910(i32*, i32*, i64*, i64, i16**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..912(i32*, i32*, i64*, i64, i32**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..914(i32*, i32*, i64*, i64, i32**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..916(i32*, i32*, i64*, i64, i32**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..918(i32*, i32*, i64*, i64, i32**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..920(i32*, i32*, i64*, i64, i64**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..922(i32*, i32*, i64*, i64, i64**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..924(i32*, i32*, i64*, i64, i64**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..926(i32*, i32*, i64*, i64, i64**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..928(i32*, i32*, i64*, i64, float**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..930(i32*, i32*, i64*, i64, float**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..932(i32*, i32*, i64*, i64, float**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..934(i32*, i32*, i64*, i64, float**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..936(i32*, i32*, i64*, i64, double**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..938(i32*, i32*, i64*, i64, double**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..940(i32*, i32*, i64*, i64, double**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..942(i32*, i32*, i64*, i64, double**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..944(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..946(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..948(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..950(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..952(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..954(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..956(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..958(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, double**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..768(i32*, i32*, i64*, i64, i8**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..770(i32*, i32*, i64*, i64, i8**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..772(i32*, i32*, i64*, i64, i8**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..774(i32*, i32*, i64*, i64, i8**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..776(i32*, i32*, i64*, i64, i16**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..778(i32*, i32*, i64*, i64, i16**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..780(i32*, i32*, i64*, i64, i16**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..782(i32*, i32*, i64*, i64, i16**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..784(i32*, i32*, i64*, i64, i32**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..786(i32*, i32*, i64*, i64, i32**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..788(i32*, i32*, i64*, i64, i32**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..790(i32*, i32*, i64*, i64, i32**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..792(i32*, i32*, i64*, i64, i64**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..794(i32*, i32*, i64*, i64, i64**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..796(i32*, i32*, i64*, i64, i64**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..798(i32*, i32*, i64*, i64, i64**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..800(i32*, i32*, i64*, i64, i8**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..802(i32*, i32*, i64*, i64, i8**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..804(i32*, i32*, i64*, i64, i8**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..806(i32*, i32*, i64*, i64, i8**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..808(i32*, i32*, i64*, i64, i16**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..810(i32*, i32*, i64*, i64, i16**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..812(i32*, i32*, i64*, i64, i16**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..814(i32*, i32*, i64*, i64, i16**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..816(i32*, i32*, i64*, i64, i32**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..818(i32*, i32*, i64*, i64, i32**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..820(i32*, i32*, i64*, i64, i32**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..822(i32*, i32*, i64*, i64, i32**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..824(i32*, i32*, i64*, i64, i64**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..826(i32*, i32*, i64*, i64, i64**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..828(i32*, i32*, i64*, i64, i64**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..830(i32*, i32*, i64*, i64, i64**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..832(i32*, i32*, i64*, i64, float**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..834(i32*, i32*, i64*, i64, float**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..836(i32*, i32*, i64*, i64, float**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..838(i32*, i32*, i64*, i64, float**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..840(i32*, i32*, i64*, i64, double**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..842(i32*, i32*, i64*, i64, double**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..844(i32*, i32*, i64*, i64, double**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..846(i32*, i32*, i64*, i64, double**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..848(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..850(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..852(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..854(i32*, i32*, i64*, i64, %"class.std::__1::complex.156"**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..856(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..858(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..860(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare void @.omp_outlined..862(i32*, i32*, i64*, i64, %"class.std::__1::complex"**, float**, %struct.st_MNumericArrayConvertData**, i32*)

declare i32 @MNumericArray_convert(%struct.st_MDataArray**, %struct.st_MDataArray*, i32, i32)

declare i32 @MNumericArray_fillFromMTensor(%struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MNumericArrayConvertData*)

declare i32 @MNumericArray_take(%struct.st_MDataArray**, %struct.st_MDataArray*, i64, i64*, i64*, i64*, i32*)

declare i32 @MNumericArray_drop(%struct.st_MDataArray**, %struct.st_MDataArray*, i64, i64*, i64*, i64*, i32*)

declare void @_ZL17rNumericArrayDropP13st_MDataArrayxxxPxS0_S1_(%struct.st_MDataArray*, i64, i64, i64, i64*, %struct.st_MDataArray*, i64*)

declare i32 @MNumericArray_join(%struct.st_MDataArray**, %struct.st_MDataArray**, i64)

declare i32 @MNumericArray_flatten(%struct.st_MDataArray**, %struct.st_MDataArray*, i64)

declare i32 @MNumericArray_clip(%struct.st_MDataArray**, %struct.st_MDataArray*, %struct.st_MDataArray*, %struct.st_MDataArray*)

declare i32 @MRawArrayType_fromString(i8*)

declare i8* @MRawArrayTypeString(i32)

declare %struct.st_MDataArray* @MRawArray_convert(%struct.st_MDataArray*, i32, i32)

declare void @KThread_initStartStopFunctions()

declare void @KThread_deleteStartStopFunctions()

declare i32 @KThread_addStartFunction(void ()*)

declare i32 @KThread_addStopFunction(void ()*)

declare void @KThread_runStartFunctions()

declare void @KThread_runStopFunction(i32)

declare void @KThread_runSomeStopFunctions(i32*, i32)

declare void @KThread_runStopFunctions()

declare i32 @getPhysicalProcessorCount()

declare i32 @sysctlbyname(i8*, i8*, i64*, i8*, i64)

declare i32 @getLogicalProcessorCount()

declare void @Delete_KThread(%struct.st_KMutex*)

declare void @KThread_Sleep(i32)

declare i32 @"\01_nanosleep"(%"struct.std::__1::pair"*, %"struct.std::__1::pair"*)

declare %struct.st_KMutex* @KThreadCreate(void (i8*)*, i8*, i32)

declare i32 @pthread_attr_init(%struct._opaque_pthread_mutex_t*)

declare i32 @pthread_attr_getschedparam(%struct._opaque_pthread_mutex_t*, %struct.sched_param*)

declare i32 @pthread_attr_setschedparam(%struct._opaque_pthread_mutex_t*, %struct.sched_param*)

declare i32 @pthread_attr_setdetachstate(%struct._opaque_pthread_mutex_t*, i32)

declare i8* @_ZL9threadRunPv(i8*)

declare i32 @pthread_attr_destroy(%struct._opaque_pthread_mutex_t*)

declare void @KThreadRawExit()

declare void @pthread_exit(i8*)

declare %struct.st_KMutex* @New_KMutex()

declare i32 @pthread_mutexattr_init(%struct._opaque_pthread_mutexattr_t*)

declare i32 @pthread_mutexattr_settype(%struct._opaque_pthread_mutexattr_t*, i32)

declare i32 @pthread_mutex_init(%struct._opaque_pthread_mutex_t*, %struct._opaque_pthread_mutexattr_t*)

declare void @Delete_KMutex(%struct.st_KMutex*)

declare i32 @pthread_mutex_destroy(%struct._opaque_pthread_mutex_t*)

declare void @KMutexLock(%struct.st_KMutex*)

declare i32 @pthread_mutex_lock(%struct._opaque_pthread_mutex_t*)

declare void @KMutexUnlock(%struct.st_KMutex*)

declare i32 @pthread_mutex_unlock(%struct._opaque_pthread_mutex_t*)

declare i32 @KMutexTryLock(%struct.st_KMutex*)

declare i32 @pthread_mutex_trylock(%struct._opaque_pthread_mutex_t*)

declare %struct.st_KMutex* @New_KCondition()

declare i32 @"\01_pthread_cond_init"(%struct._opaque_pthread_cond_t*, %struct._opaque_pthread_mutexattr_t*)

declare void @Delete_KCondition(%struct.st_KMutex*)

declare i32 @pthread_cond_destroy(%struct._opaque_pthread_cond_t*)

declare void @KConditionWait(%struct.st_KMutex*, %struct.st_KMutex*)

declare i32 @"\01_pthread_cond_wait"(%struct._opaque_pthread_cond_t*, %struct._opaque_pthread_mutex_t*)

declare i32 @KConditionTimedWait(%struct.st_KMutex*, %struct.st_KMutex*, i64)

declare i32 @gettimeofday(%struct.timeval*, i8*)

declare i32 @"\01_pthread_cond_timedwait"(%struct._opaque_pthread_cond_t*, %struct._opaque_pthread_mutex_t*, %"struct.std::__1::pair"*)

declare void @KConditionSignal(%struct.st_KMutex*)

declare i32 @pthread_cond_signal(%struct._opaque_pthread_cond_t*)

declare i32 @KThreadWaitForExit(%struct.st_KMutex*)

declare i32 @"\01_pthread_join"(%struct._opaque_pthread_t*, i8**)

declare %struct.st_KMutex* @New_KThreadLocalStorage()

declare i32 @pthread_key_create(i64*, void (i8*)*)

declare void @Delete_KThreadLocalStorage(%struct.st_KMutex*)

declare i32 @pthread_key_delete(i64)

declare void @KThreadLocalStorageSet(%struct.st_KMutex*, i8*)

declare i8* @KThreadLocalStorageGet(%struct.st_KMutex*)

declare i8* @pthread_getspecific(i64)

declare i32 @getProcessPriorityClass()

declare i32* @__error()

declare i32 @getpriority(i32, i32)

declare i32 @setProcessPriorityClass(i32)

declare i32 @setpriority(i32, i32, i32)

declare i64 @KThreadID()

declare %struct._opaque_pthread_t* @pthread_self()

declare void @_Z10InitRandomi(i32)

declare %struct.RandomGenerator_struct* @RandomGenerator_newGeneric()

declare i32 @Generic_randomMIntegers(i64*, i64, i64, i64, %struct.RandomGenerator_struct*)

declare i32 @Generic_randomMReals(double*, i64, double, double, %struct.RandomGenerator_struct*)

declare %struct.RandomDistributionFunctionArray_struct* @RandomDistributionFunctionArray_copy(%struct.RandomDistributionFunctionArray_struct*)

declare void @RandomDistributionFunctionArray_delete(%struct.RandomDistributionFunctionArray_struct*)

declare %struct.RandomGenerator_struct* @RandomGenerator_copyGeneric(%struct.RandomGenerator_struct*)

declare void @RandomGenerator_deleteGeneric(%struct.RandomGenerator_struct*)

declare %struct.RandomGeneratorMethodData_struct* @RandomGeneratorMethodData_new(i64, i8*)

declare void @RandomGeneratorMethodData_delete(%struct.RandomGeneratorMethodData_struct*)

declare %struct.RandomGeneratorMethod_struct* @RandomGeneratorMethod_new(i8*)

declare %struct.RandomGenerator_struct* @RandomGeneratorMethod_getCurrent(%struct.RandomGeneratorMethod_struct*)

declare void @RandomGeneratorMethod_setCurrent(%struct.RandomGeneratorMethod_struct*, %struct.RandomGenerator_struct*)

declare i64 @RandomGeneratorMethod_find(i8*)

declare %struct.RandomGeneratorMethod_struct* @RandomGeneratorMethod_get(i64)

declare void @RandomGeneratorMethod_addToMethodList(%struct.RandomGeneratorMethod_struct*)

declare void @RandomGeneratorMethod_add(i8*, %struct.RandomGenerator_struct* (i8*)*, %struct.RandomGenerator_struct* (%struct.RandomGenerator_struct*)*, void (%struct.RandomGenerator_struct*)*)

declare void @RandomGenerator_refIncr(%struct.RandomGenerator_struct*)

declare void @RandomGenerator_free(%struct.RandomGenerator_struct*)

declare void @RandomGenerator_refDecr(%struct.RandomGenerator_struct*)

declare %struct.RandomGenerator_struct* @RandomGenerator_clone(%struct.RandomGenerator_struct*)

declare void @CurrentRandomGenerator_set(%struct.RandomGenerator_struct*)

declare %struct.RandomGenerator_struct* @CurrentRandomGenerator_get()

declare %struct.RandomGenerator_struct* @GetCurrentRandomGenerator()

declare %struct.random_state_entry_struct* @RandomStateEntry_new()

declare %struct.random_state_entry_struct* @RandomStateEntry_add(%struct.random_state_entry_struct*, %struct.RandomGenerator_struct*)

declare void @RandomStateEntry_delete(%struct.random_state_entry_struct*)

declare void @RandomStateEntry_seed(%struct.random_state_entry_struct*, i64*, i64)

declare i64 @"\01_clock"()

declare %struct.RandomState_struct* @RandomState_new()

declare void @RandomState_count(%struct.RandomState_struct*)

declare void @_ZL23RandomState_countThreadPvP17st_ParallelThread(i8*, %struct.st_ParallelThread*)

declare void @RandomState_discount(%struct.RandomState_struct*)

declare void @_ZL26RandomState_discountThreadPvP17st_ParallelThread(i8*, %struct.st_ParallelThread*)

declare void @RandomState_delete(%struct.RandomState_struct*)

declare void @_ZL24RandomState_deleteThreadPvP17st_ParallelThread(i8*, %struct.st_ParallelThread*)

declare i64 @GetParallelGeneratorIndex()

declare void @ClearParallelRandomGenerators()

declare void @_ZL37ClearThreadRandomGeneratorsIterateFunPvP17st_ParallelThread(i8*, %struct.st_ParallelThread*)

declare void @_ZL27ClearThreadRandomGeneratorsv()

declare %struct.RandomState_struct* @RandomState_fromSpecification(i8*, i8*, i64*, i64)

declare %struct.RandomState_struct* @_ZL19RandomState_fromAllPyx(i64*, i64)

declare %struct.RandomState_struct* @GetRandomState(i8*)

declare void @RestoreRandomState(%struct.RandomState_struct*)

declare void @InitializeRandomThreads()

declare void @DeinitializeRandomThreads()

declare void @_Z24init_rtl_randomgeneratori(i32)

declare %struct.buffer_data_struct* @BufferData_new(i64)

declare %struct.buffer_data_struct* @BufferData_copy(%struct.buffer_data_struct*)

declare void @BufferData_clear(%struct.buffer_data_struct*)

declare void @BufferData_delete(%struct.buffer_data_struct*)

declare %struct.RandomBuffer_struct* @RandomBuffer_new(i32 (i64*, i64*, i64*, %struct.RandomGenerator_struct*)*, i64)

declare i64 @_ZL20RandomBuffer_getBitsxP22RandomGenerator_struct(i64, %struct.RandomGenerator_struct*)

declare i32 @_ZL25RandomBuffer_randomBigitsPyxP22RandomGenerator_struct(i64*, i64, %struct.RandomGenerator_struct*)

declare i32 @_ZL28RandomBuffer_randomMIntegersPxxxxP22RandomGenerator_struct(i64*, i64, i64, i64, %struct.RandomGenerator_struct*)

declare i32 @_ZL25RandomBuffer_randomMRealsPdxddP22RandomGenerator_struct(double*, i64, double, double, %struct.RandomGenerator_struct*)

declare void @_ZL20RandomBuffer_setSeedP22RandomGenerator_structPyx(%struct.RandomGenerator_struct*, i64*, i64)

declare %struct.RandomBuffer_struct* @RandomBuffer_copy(%struct.RandomBuffer_struct*)

declare void @RandomBuffer_delete(%struct.RandomBuffer_struct*)

declare i32 @UniformRandomMReals(double*, i64, double, double)

declare i32 @UniformRandomMComplexes(%"class.std::__1::complex"*, i64, double, double, double, double)

declare i32 @UniformRandomMIntegers(i64*, i64, i64, i64)

declare i32 @AccumulateWeightMTensor(%struct.st_MDataArray**, %struct.st_MDataArray*)

declare i32 @RandomChoicePositionsM(i64*, i64, double*, i64, %struct.RandomGenerator_struct*)

declare i32 @RandomChoices(i64*, i64, i64, %struct.st_MDataArray*)

declare %struct.st_hashtable* @PositionHashTable_new(i64)

declare i64 @_ZL15PositionHashFunP13st_hash_entry(%struct.st_hash_entry*)

declare void @_ZL15PositionHashSetP13st_hash_entryS0_(%struct.st_hash_entry*, %struct.st_hash_entry*)

declare i32 @_ZL17PositionHashEqualP13st_hash_entryS0_(%struct.st_hash_entry*, %struct.st_hash_entry*)

declare void @_ZL15PositionHashNewP13st_hash_entry(%struct.st_hash_entry*)

declare void @_ZL18PositionHashDeleteP13st_hash_entry(%struct.st_hash_entry*)

declare i32 @SimpleRandomSamplePositionsM(i64*, i64, i64, %struct.RandomGenerator_struct*)

declare i32 @RandomSamplePositionsM(i64*, i64, %struct.st_MDataArray*, %struct.RandomGenerator_struct*)

declare %struct.PositionTree_struct* @_ZL16MakePositionTreexPxPd(i64, i64*, double*)

declare i64 @_ZL20ChooseSamplePositionP19PositionTree_structd(%struct.PositionTree_struct*, double)

declare void @_ZL19PositionTree_deleteP19PositionTree_struct(%struct.PositionTree_struct*)

declare i32 @RandomSamples(i64*, i64, i64, %struct.st_MDataArray*)

declare i32 @AcceptanceRejectionMRealVector(double*, i64, double*, double*, i64, double*, i32 (double*, double*)*)

declare i32 @BetaMRealVector(double*, i64, double, double)

; Function Attrs: nounwind readnone speculatable
declare <2 x double> @llvm.pow.v2f64(<2 x double>, <2 x double>) #0

declare i32 @CA5State_checkParameters(i64, i64, i64)

declare %struct.RandomGenerator_struct* @CA5BufferState_alloc(i8*)

declare void @_ZL11CA5_setSeedP22RandomGenerator_structPyx(%struct.RandomGenerator_struct*, i64*, i64)

declare i32 @_ZL19CA5_fillBufferArrayPyPxS0_P22RandomGenerator_struct(i64*, i64*, i64*, %struct.RandomGenerator_struct*)

declare %struct.RandomGenerator_struct* @CA5BufferState_clone(%struct.RandomGenerator_struct*)

declare void @CA5BufferState_free(%struct.RandomGenerator_struct*)

declare void @_Z18init_rtl_randomCA5i(i32)

declare i32 @MKLLookupMethod(i8*)

declare i64 @MKLMethodNames(i8***)

declare %struct.RandomGenerator_struct* @MKLBufferState_alloc(i8*)

declare i32 @_ZL19MKL_fillBufferArrayPyPxS0_P22RandomGenerator_struct(i64*, i64*, i64*, %struct.RandomGenerator_struct*)

declare i32 @_ZL30MKLBufferState_randomMIntegersPxxxxP22RandomGenerator_struct(i64*, i64, i64, i64, %struct.RandomGenerator_struct*)

declare i32 @_ZL28MKLBufferState_randomDoublesPdxddP22RandomGenerator_struct(double*, i64, double, double, %struct.RandomGenerator_struct*)

declare void @_ZL16MKLState_setSeedP22RandomGenerator_structPyx(%struct.RandomGenerator_struct*, i64*, i64)

declare i32 @vslGetBrngProperties(i32, %struct._VSLBRngProperties*)

declare i32 @vslDeleteStream(i8**)

declare i32 @vslNewStream(i8**, i64, i64)

declare i32 @vslSkipAheadStream(i8*, i64)

declare i32 @viRngUniformBits(i64, i8*, i64, i32*)

declare i32 @vdRngUniform(i64, i8*, i64, double*, double, double)

declare i32 @viRngUniform(i64, i8*, i64, i32*, i32, i32)

declare %struct.RandomGenerator_struct* @MKLBufferState_clone(%struct.RandomGenerator_struct*)

declare i32 @vslCopyStream(i8**, i8*)

declare void @MKLBufferState_free(%struct.RandomGenerator_struct*)

declare i32 @MKLBufferState_randomNormals(double*, i64, double, double, %struct.RandomGenerator_struct*)

declare i32 @vdRngGaussian(i64, i8*, i64, double*, double, double)

declare void @_Z18init_rtl_randomMKLi(i32)

declare %struct.RandomGenerator_struct* @MersenneTwister_alloc(i8*)

declare i32 @_ZL31MersenneTwister_fillBufferArrayPyPxS0_P22RandomGenerator_struct(i64*, i64*, i64*, %struct.RandomGenerator_struct*)

declare void @_ZL23MersenneTwister_setSeedP22RandomGenerator_structPyx(%struct.RandomGenerator_struct*, i64*, i64)

declare %struct.RandomGenerator_struct* @MersenneTwister_clone(%struct.RandomGenerator_struct*)

declare void @MersenneTwister_free(%struct.RandomGenerator_struct*)

declare i64 @PMT32_getNextIndex()

declare i32 @PMT32_checkParameters(i64)

declare %struct.RandomGenerator_struct* @PMT32_alloc(i8*)

declare void @_ZL13PMT32_setSeedP22RandomGenerator_structPyx(%struct.RandomGenerator_struct*, i64*, i64)

declare i32 @_ZL21PMT32_fillBufferArrayPyPxS0_P22RandomGenerator_struct(i64*, i64*, i64*, %struct.RandomGenerator_struct*)

declare void @_ZL14PMT32_generatePjP17PMT32State_struct(i32*, %struct.PMT32State_struct*)

declare %struct.RandomGenerator_struct* @PMT32_clone(%struct.RandomGenerator_struct*)

declare void @PMT32_free(%struct.RandomGenerator_struct*)

declare void @_Z17init_rtl_randomMTi(i32)

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind readnone }
