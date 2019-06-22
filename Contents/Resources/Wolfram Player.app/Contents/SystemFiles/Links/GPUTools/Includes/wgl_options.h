/*********************************************************************//**
* @file
*
* @section LICENCE
*
*              Mathematica source file
*
*  Copyright 1986 through 2011 by Wolfram Research Inc.
*
* This material contains trade secrets and may be registered with the
* U.S. Copyright Office as an unpublished work, pursuant to Title 17,
* U.S. Code, Section 408.  Unauthorized copying, adaptation, distribution
* or display is prohibited.
*
* @section DESCRIPTION
*
*
*
* $Id$
************************************************************************/

#ifndef __WGL_OPTIONS_H__
#define __WGL_OPTIONS_H__

#if defined(_DEBUG) && !defined(DEBUG)
#define DEBUG
#endif /* defined(_DEBUG) && !defined(DEBUG) */

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#endif

#ifndef UNICODE
#define UNICODE									1
#endif /* UNICODE */

#define WGL_STATE_STRING_BUFFER_SIZE			4096

#define WGL_Tree_Delta_Max						1

#define WGL_MAXIMUM_STRINGLENGTH                1024

#define WGL_TENSOR_HASH_FUNCTION				Hash_PJW
#define WGL_SIZEOF_ERROR_MESSAGE				4096
#define WGL_BUILD_LOG_BUFFER_SIZE				4096

#define WGL_MAXIMUM_USABLE_MEMORY_PRECENTAGE		0.8
#define WGL_MAXIMUM_USABLE_SHARED_MEMORY_PRECENTAGE	0.5

#define MAXIMUM_QUERY_INFORMATION_STRING_LENGTH 1024

#ifdef CONFIG_USE_CUDA
#define CONFIG_USE_CUDA_FFT						1
#define CONFIG_USE_CUDA_IMAGE_PROCESSING		1
//#define CONFIG_USE_CUSPARSE					1

#endif /* CONFIG_USE_CUDA */

#define WGL_MEMORY_POOL_SIZE					256
#define WGL_MEMORY_POOL_BUFFER_TOLERANCE		(256*(1L<<10)) /* 256 kilo bytes */
#define WGL_MEMORY_POOL_MIN_ELEMENT_SIZE		WGL_MEMORY_POOL_BUFFER_TOLERANCE

#define WGL_IMAGEPROCESSING_KERNEL_MAX_DIM		50
//#define CONFIG_NO_SHARED_MEMORY				1

//#define CONFIG_ENABLE_ASYNC					1

#define WGL_STATE_USER_DATA_SIZE				16

#endif /* __WGL_OPTIONS_H__ */



