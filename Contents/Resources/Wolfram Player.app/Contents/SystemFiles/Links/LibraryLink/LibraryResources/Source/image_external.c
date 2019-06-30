/* Include required header */
#include "WolframLibrary.h"
#include "WolframImageLibrary.h"
#include "cv.h"
#include "highgui.h"
#include "imgproc_c.h"
#include "libraw.h"

/* Return the version of Library Link */
DLLEXPORT mint WolframLibrary_getVersion( ) {
	return WolframLibraryVersion;
}

/* Initialize Library */
DLLEXPORT int WolframLibrary_initialize(WolframLibraryData libData) {
	return LIBRARY_NO_ERROR;
}

/* Uninitialize Library */
DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData libData) {
	return;
}

DLLEXPORT int opencv_dilate(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res)
{
	mint dims[3], w, h, i, j;
	IplImage* src = 0;
	IplImage* dst = 0;
	raw_t_bit* src_data_bit = 0;
	raw_t_bit* dst_data_bit = 0;
	raw_t_ubit8* src_data_byte = 0;
	raw_t_ubit8* dst_data_byte = 0;
	raw_t_ubit16* src_data_bit16 = 0;
	raw_t_ubit16* dst_data_bit16 = 0;
	raw_t_real32* src_data_real32 = 0;
	raw_t_real32* dst_data_real32 = 0;
	IplConvKernel* element = 0;
	MImage image_in, image_out = 0;
	int radius;
	int err = 0;
	int type = 0;
	WolframImageLibrary_Functions imgFuns = libData->imageLibraryFunctions;
	
	if (Argc < 2) {
		return LIBRARY_FUNCTION_ERROR;
	}
	
	image_in = MArgument_getMImage(Args[0]);
	if(imgFuns->MImage_getRank(image_in) == 3) return LIBRARY_FUNCTION_ERROR;
	if(imgFuns->MImage_getChannels(image_in) != 1) return LIBRARY_FUNCTION_ERROR;
	radius = MArgument_getInteger(Args[1]);
	if(radius < 1) return LIBRARY_FUNCTION_ERROR;
	err = imgFuns->MImage_clone(image_in, &image_out);
	if (err) return LIBRARY_FUNCTION_ERROR;
	
	type = imgFuns->MImage_getDataType(image_in);
	h = imgFuns->MImage_getRowCount(image_in);
	w = imgFuns->MImage_getColumnCount(image_in);
	
	element = cvCreateStructuringElementEx(2*radius+1,2*radius+1, radius, radius, CV_SHAPE_RECT, 0);

	switch(type) {
		case MImage_Type_Bit: 
		{
			raw_t_bit* data_in = imgFuns->MImage_getBitData(image_in);
			raw_t_bit* data_out = imgFuns->MImage_getBitData(image_out);
			if (!data_in || !data_out) {
				err = LIBRARY_FUNCTION_ERROR;
				goto cleanup;
			}
			src = cvCreateImage( cvSize(w, h), IPL_DEPTH_1U, 1);
			dst = cvCreateImage( cvSize(w, h), IPL_DEPTH_1U, 1);
			src_data_bit = src->imageData;
			dst_data_bit = dst->imageData;
			for (i = 0; i < h; i++) {
				for (j = 0; j < w; j++) {
					(src_data_bit + i*src->widthStep)[j] = data_in[i*w+j];
				}
			}
			cvDilate(src, dst, element, 1);
			for (i = 0; i < h; i++) {
				for (j = 0; j < w; j++) {
					data_out[i*w+j] = (dst_data_bit + i*dst->widthStep)[j];
				}
			}
			break;
		}
		case MImage_Type_Bit8:
		{
		    raw_t_ubit8* data_in = imgFuns->MImage_getByteData(image_in);
			raw_t_ubit8* data_out = imgFuns->MImage_getByteData(image_out);
			if (!data_in || !data_out) {
				err = LIBRARY_FUNCTION_ERROR;
				goto cleanup;
			}
			src = cvCreateImage( cvSize(w, h), IPL_DEPTH_8U, 1);
			dst = cvCreateImage( cvSize(w, h), IPL_DEPTH_8U, 1);
			src_data_byte = src->imageData;
			dst_data_byte = dst->imageData;
			for (i = 0; i < h; i++) {
				for (j = 0; j < w; j++) {
					(src_data_byte + i*src->widthStep)[j] = data_in[i*w+j];
				}
			}
			cvDilate(src, dst, element, 1);
			for (i = 0; i < h; i++) {
				for (j = 0; j < w; j++) {
					data_out[i*w+j] = (dst_data_byte + i*dst->widthStep)[j];
				}
			}
			break;
		}
		case MImage_Type_Bit16:
		{
		    raw_t_ubit16* data_in = imgFuns->MImage_getBit16Data(image_in);
			raw_t_ubit16* data_out = imgFuns->MImage_getBit16Data(image_out);
			if (!data_in || !data_out) {
				err = LIBRARY_FUNCTION_ERROR;
				goto cleanup;
			}
			src = cvCreateImage( cvSize(w, h), IPL_DEPTH_16U, 1);
			dst = cvCreateImage( cvSize(w, h), IPL_DEPTH_16U, 1);
			src_data_bit16 = src->imageData;
			dst_data_bit16 = dst->imageData;
			for (i = 0; i < h; i++) {
				for (j = 0; j < w; j++) {
					(src_data_bit16 + i*src->widthStep)[j] = data_in[i*w+j];
				}
			}
			cvDilate(src, dst, element, 1);
			for (i = 0; i < h; i++) {
				for (j = 0; j < w; j++) {
					data_out[i*w+j] = (dst_data_bit16 + i*dst->widthStep)[j];
				}
			}
			break;
		}
		case MImage_Type_Real32:
		{
		    raw_t_real32* data_in = imgFuns->MImage_getReal32Data(image_in);
			raw_t_real32* data_out = imgFuns->MImage_getReal32Data(image_out);
			if (!data_in || !data_out) {
				err = LIBRARY_FUNCTION_ERROR;
				goto cleanup;
			}
			src = cvCreateImage( cvSize(w, h), IPL_DEPTH_32F, 1);
			dst = cvCreateImage( cvSize(w, h), IPL_DEPTH_32F, 1);
			src_data_real32 = src->imageData;
			dst_data_real32 = dst->imageData;
			for (i = 0; i < h; i++) {
				for (j = 0; j < w; j++) {
					(src_data_real32 + i*src->widthStep)[j] = data_in[i*w+j];
				}
			}
			cvDilate(src, dst, element, 1);
			for (i = 0; i < h; i++) {
				for (j = 0; j < w; j++) {
					data_out[i*w+j] = (dst_data_real32 + i*dst->widthStep)[j];
				}
			}
			break;
		}
		case MImage_Type_Real:
		{
		    raw_t_real64* data_in = imgFuns->MImage_getRealData(image_in);
			raw_t_real64* data_out = imgFuns->MImage_getRealData(image_out);
			if (!data_in || !data_out) {
				err = LIBRARY_FUNCTION_ERROR;
				goto cleanup;
			}
			src = cvCreateImage( cvSize(w, h), IPL_DEPTH_32F, 1);
			dst = cvCreateImage( cvSize(w, h), IPL_DEPTH_32F, 1);
			src_data_real32 = src->imageData;
			dst_data_real32 = dst->imageData;
			for (i = 0; i < h; i++) {
				for (j = 0; j < w; j++) {
					(src_data_real32 + i*src->widthStep)[j] = (raw_t_real32)data_in[i*w+j];
				}
			}
			cvDilate(src, dst, element, 1);
			for (i = 0; i < h; i++) {
				for (j = 0; j < w; j++) {
					data_out[i*w+j] = (dst_data_real32 + i*dst->widthStep)[j];
				}
			}
			break;
		}
		default:
		return LIBRARY_FUNCTION_ERROR;
	}

cleanup:
	if(src) cvReleaseImage( &src );
	if(dst) cvReleaseImage( &dst );
	if(element) cvReleaseStructuringElement(&element);
	if(err == 0) {
		MArgument_setMImage(res, image_out);
	}
	else {
		if(image_out) imgFuns->MImage_free(image_out);
	}
	return err;
}

EXTERN_C DLLEXPORT int read_raw_image(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res) {
	int err;
	int check;
	MImage out;
	char * file;
	libraw_data_t *iprc = libraw_init(0);
	libraw_processed_image_t * img;
	WolframImageLibrary_Functions imgFuns = libData->imageLibraryFunctions;

	err = LIBRARY_FUNCTION_ERROR;
	file = MArgument_getUTF8String(Args[0]);

	libraw_open_file(iprc, file);
	libraw_unpack(iprc);

	iprc->params.output_bps = 8;

	check = libraw_dcraw_process(iprc);
	if (check != LIBRAW_SUCCESS) goto cleanup;

	img = libraw_dcraw_make_mem_image(iprc, &check);
	if (img == NULL) goto cleanup;
	if (img->type != LIBRAW_IMAGE_BITMAP || img->colors != 3) goto cleanup;
	
	if (img->bits == 16) {
		raw_t_ubit16 * raw_data = (raw_t_ubit16*)img->data;
		imgFuns->MImage_new2D(img->width, img->height, 3, MImage_Type_Bit16, MImage_CS_RGB, 1, &out);
		memcpy(imgFuns->MImage_getBit16Data(out), raw_data, img->width * img->height * 3 * sizeof(raw_t_ubit16));
	} else if (img->bits == 8) {
		raw_t_ubit8 * raw_data = (raw_t_ubit8*)img->data;
		imgFuns->MImage_new2D(img->width, img->height, 3, MImage_Type_Bit8, MImage_CS_RGB, 1, &out);
		memcpy(imgFuns->MImage_getByteData(out), raw_data, img->width * img->height * 3 * sizeof(raw_t_ubit8));
	} else {
		goto cleanup;
	}
	
	MArgument_setMImage(res, out);
	err = LIBRARY_NO_ERROR;

cleanup:
	libData->UTF8String_disown(file);
	libraw_dcraw_clear_mem(img);
	return err;
}
