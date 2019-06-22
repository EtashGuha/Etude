
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <WolframLibrary.h>
#include <WolframImageLibrary.h>

extern "C" {
	#include <libavcodec/avcodec.h>
	#include <libavformat/avformat.h>
	#include <libswscale/swscale.h>
}

#ifdef _WIN32
#include "unordered_map"
#else /* _WIN32 */
#include "tr1/unordered_map"
#endif /* _WIN32 */

namespace std {
	using namespace tr1;
}

typedef struct st_VideoHandle {
	AVFormatContext * avFormatContext;
	AVCodecContext * avCodecContext;
	AVCodec * avCodec;
	AVFrame * avFrameRaw, *avFrameData;
	uint8_t * videoBuffer;
	struct SwsContext * swsContext;
	int videoStreamIndex, width, height;
	AVPixelFormat format;
} * VideoHandle;


typedef std::unordered_map<mint, VideoHandle *> VideoHash_t;

static VideoHash_t * VideoHash;

static void ffmpegInitialize();
static int videoHandleNew(const char * file, int width, int height, AVPixelFormat format, VideoHandle * vh);
static void ffmpegSeekFrame(VideoHandle * vh, mint n, mint flag);
static void ffmpegSeekSeconds(VideoHandle * vh, double n, mint flag);
static void ffmpegNextFrame(VideoHandle * vh);
static void ffmpegSkipFrame(VideoHandle * vh);
static void videoHandleFree(VideoHandle * vh);

void ffmpegInitialize() {
	av_register_all();
	return;
}

int videoHandleNew(const char * file, int width, int height, AVPixelFormat format, VideoHandle * vh) {
	unsigned int ii = 0;
	AVDictionary *optionsDict = NULL;

	*vh = (VideoHandle)malloc(sizeof(struct st_VideoHandle));
	if (*vh == NULL) {
		return LIBRARY_FUNCTION_ERROR;
	}

	(*vh)->format = format;

	(*vh)->avFormatContext = NULL;
	if (avformat_open_input(&((*vh)->avFormatContext), file, NULL, NULL)) {
		videoHandleFree(vh);
		return LIBRARY_FUNCTION_ERROR;
	}

	if (avformat_find_stream_info((*vh)->avFormatContext, NULL) < 0) {
		videoHandleFree(vh);
		return LIBRARY_FUNCTION_ERROR;
	}

	do {
		if ((*vh)->avFormatContext->streams[ii]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
			(*vh)->videoStreamIndex = ii;
			break;
		}
		else {
			ii++;
		}
	} while (ii < (*vh)->avFormatContext->nb_streams);


	(*vh)->avCodecContext = (*vh)->avFormatContext->streams[(*vh)->videoStreamIndex]->codec;
	(*vh)->width = width > 0 ? width : (*vh)->avCodecContext->width;
	(*vh)->height = height > 0 ? height : (*vh)->avCodecContext->height;
	(*vh)->avCodec = avcodec_find_decoder((*vh)->avCodecContext->codec_id);


	if ((*vh)->avCodec == NULL || avcodec_open2((*vh)->avCodecContext, (*vh)->avCodec, &optionsDict) < 0) {
		videoHandleFree(vh);
		return LIBRARY_FUNCTION_ERROR;
	}

	/* Frame rate fix for some codecs */
	if ((*vh)->avCodecContext->time_base.num > 1000 && (*vh)->avCodecContext->time_base.den == 1)
		(*vh)->avCodecContext->time_base.den = 1000;

	(*vh)->avFrameRaw = av_frame_alloc();
	(*vh)->avFrameData = av_frame_alloc();
	
	if ((*vh)->avFrameRaw == NULL || (*vh)->avFrameData == NULL) {
		videoHandleFree(vh);
		return LIBRARY_FUNCTION_ERROR;
	}

	(*vh)->videoBuffer = (uint8_t *)malloc(avpicture_get_size((*vh)->format, (*vh)->avCodecContext->width, (*vh)->avCodecContext->height));


	avpicture_fill((AVPicture *)(*vh)->avFrameData, (*vh)->videoBuffer, (*vh)->format, (*vh)->avCodecContext->width, (*vh)->avCodecContext->height);


	(*vh)->swsContext = sws_getContext((*vh)->avCodecContext->width, (*vh)->avCodecContext->height, (*vh)->avCodecContext->pix_fmt,
		(*vh)->width, (*vh)->height, (*vh)->format, SWS_BICUBIC, NULL, NULL, NULL);

	if ((*vh)->swsContext == NULL) {
		videoHandleFree(vh);
		return LIBRARY_FUNCTION_ERROR;
	}

	return LIBRARY_NO_ERROR;
}

void ffmpegSeekFrame(VideoHandle * vh, mint n, mint flag) {
	switch (flag) {
	case 1:
		av_seek_frame((*vh)->avFormatContext, (*vh)->videoStreamIndex, n, AVSEEK_FLAG_ANY);
	case 2:
		av_seek_frame((*vh)->avFormatContext, (*vh)->videoStreamIndex, n, AVSEEK_FLAG_BACKWARD);
	default:
		av_seek_frame((*vh)->avFormatContext, (*vh)->videoStreamIndex, n, AVSEEK_FLAG_BYTE);
	}
}

void ffmpegSeekSeconds(VideoHandle * vh, double n, mint flag) {
	ffmpegSeekFrame(vh, (mint) (n * AV_TIME_BASE), flag);
}

void ffmpegSkipFrame(VideoHandle * vh) {
	AVPacket packet;
	int status = -1;

	while (av_read_frame((*vh)->avFormatContext, &packet) >= 0) {
		if (packet.stream_index == (*vh)->videoStreamIndex) {
			avcodec_decode_video2((*vh)->avCodecContext, (*vh)->avFrameRaw, &status, &packet);
			if (status) {
				return;
			}
		}
	}
}

void ffmpegNextFrame(VideoHandle * vh) {

	
	ffmpegSkipFrame(vh);

	sws_scale((*vh)->swsContext, (const uint8_t * const *)(*vh)->avFrameRaw->data, (*vh)->avFrameRaw->linesize,
		0, (*vh)->avCodecContext->height, (*vh)->avFrameData->data, (*vh)->avFrameData->linesize);

	return;
}

void videoHandleFree(VideoHandle * vh) {
	if (vh == NULL || *vh == NULL) {
		return;
	}
	if ((*vh)->swsContext != NULL)
		sws_freeContext((*vh)->swsContext);

	if ((*vh)->avFrameData != NULL)
		av_free((*vh)->avFrameData);
	if ((*vh)->avFrameRaw != NULL)
		av_free((*vh)->avFrameRaw);

	if ((*vh)->avCodecContext)
		avcodec_close((*vh)->avCodecContext);
	if ((*vh)->avFormatContext)
		avformat_close_input(&((*vh)->avFormatContext));

	if ((*vh)->videoBuffer != NULL)
		free((*vh)->videoBuffer);
	free(*vh);

	return;
}

static int videoHandle_toMImage(WolframImageLibrary_Functions imgFuns, VideoHandle * vh, MImage * img) {
	int err = LIBRARY_TYPE_ERROR;

	if (vh == NULL || *vh == NULL) {
		return LIBRARY_MEMORY_ERROR;
	}

	if ((*vh)->format == AV_PIX_FMT_RGBA) {
		err = imgFuns->MImage_new2D((*vh)->width, (*vh)->height, 4, MImage_Type_Bit8, MImage_CS_RGB, True, img);
		if (err == LIBRARY_NO_ERROR) {
			raw_t_ubit8 * imgData = imgFuns->MImage_getByteData(*img);
			memcpy(imgData, (*vh)->videoBuffer, 4 * (*vh)->width*(*vh)->height*sizeof(raw_t_ubit8));
		}
	}

	return err;
}


EXTERN_C DLLEXPORT void manageInstance(WolframLibraryData libData, mbool mode, mint id) {
	
	if (mode == False) {
		VideoHandle * vh = (VideoHandle *)malloc(sizeof(VideoHandle));
		(*VideoHash)[id] = vh;
		*vh = NULL;
	}
	else {
		VideoHandle * vh;
		VideoHash_t::iterator iter = VideoHash->find(id);
		if (iter != VideoHash->end()) {
			vh = iter->second;
			videoHandleFree(vh);
			if (vh != NULL) {
				free(vh);
			}
		}
	}
	return;
}


EXTERN_C DLLEXPORT int releaseInstance(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument res) {
	mint id;

	if (Argc != 1) {
		return LIBRARY_FUNCTION_ERROR;
	}

	id = MArgument_getInteger(Args[0]);
	return libData->releaseManagedLibraryExpression("VideoInstance", id);
}


EXTERN_C DLLEXPORT mint WolframLibrary_getVersion() {
	return WolframLibraryVersion;
}

EXTERN_C DLLEXPORT int WolframLibrary_initialize(WolframLibraryData libData) {

	VideoHash = new VideoHash_t();

	ffmpegInitialize();

	libData->registerLibraryExpressionManager("VideoInstance", manageInstance);

	return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData libData) {
	libData->unregisterLibraryExpressionManager("VideoInstance");

	delete VideoHash;

	return;
}

EXTERN_C DLLEXPORT int oCreate(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res) {
	char * pth;
	mint id;
	int err;
	VideoHandle * vh;
	VideoHash_t::iterator iter;

	id = MArgument_getInteger(Args[0]);
	pth = MArgument_getUTF8String(Args[1]);


	iter = VideoHash->find(id);
	if (iter == VideoHash->end()) {
		goto cleanup;
	}

	vh = iter->second;

	err = videoHandleNew(pth, 0, 0, PIX_FMT_RGBA, vh);
	if (err != LIBRARY_NO_ERROR)
		goto cleanup;

	err = LIBRARY_NO_ERROR;

cleanup:
	libData->UTF8String_disown(pth);

	return err;
}

EXTERN_C DLLEXPORT int oSeekFrame(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res) {
	int err;
	mint id;
	mint n;
	mint flag;
	MImage img;
	VideoHandle * vh;
	VideoHash_t::iterator iter;
	WolframImageLibrary_Functions imgFuns = libData->imageLibraryFunctions;

	id = MArgument_getInteger(Args[0]);
	n = MArgument_getInteger(Args[0]);
	flag = MArgument_getInteger(Args[1]);

	iter = VideoHash->find(id);
	if (iter == VideoHash->end()) {
		return LIBRARY_MEMORY_ERROR;
	}
	
	vh = iter->second;

	ffmpegSeekFrame(vh, n, flag);
	ffmpegNextFrame(vh);

	err = videoHandle_toMImage(imgFuns, vh, &img);

	if (err == LIBRARY_NO_ERROR)
		MArgument_setMImage(Res, img);

	return err;
}

EXTERN_C DLLEXPORT int oSeekSeconds(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res) {
	int err;
	mint id;
	double n;
	mint flag;
	MImage img;
	VideoHandle * vh;
	VideoHash_t::iterator iter;
	WolframImageLibrary_Functions imgFuns = libData->imageLibraryFunctions;

	id = MArgument_getInteger(Args[0]);
	n = MArgument_getReal(Args[1]);
	flag = MArgument_getInteger(Args[2]);

	iter = VideoHash->find(id);
	if (iter == VideoHash->end()) {
		return LIBRARY_MEMORY_ERROR;
	}

	vh = iter->second;

	ffmpegSeekSeconds(vh, n, flag);
	ffmpegNextFrame(vh);

	err = videoHandle_toMImage(imgFuns, vh, &img);

	if (err == LIBRARY_NO_ERROR)
		MArgument_setMImage(Res, img);

	return err;
}

EXTERN_C DLLEXPORT int oSkipFrame(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res) {
	int err;
	mint id;
	MImage img;
	VideoHandle * vh;
	VideoHash_t::iterator iter;
	WolframImageLibrary_Functions imgFuns = libData->imageLibraryFunctions;

	id = MArgument_getInteger(Args[0]);

	iter = VideoHash->find(id);
	if (iter == VideoHash->end()) {
		return LIBRARY_MEMORY_ERROR;
	}

	vh = iter->second;

	ffmpegSkipFrame(vh);
	ffmpegNextFrame(vh);

	err = videoHandle_toMImage(imgFuns, vh, &img);

	if (err == LIBRARY_NO_ERROR)
		MArgument_setMImage(Res, img);

	return err;
}

EXTERN_C DLLEXPORT int oNextFrame(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res) {
	int err;
	mint id;
	MImage img;
	VideoHandle * vh;
	VideoHash_t::iterator iter;
	WolframImageLibrary_Functions imgFuns = libData->imageLibraryFunctions;

	id = MArgument_getInteger(Args[0]);

	iter = VideoHash->find(id);
	if (iter == VideoHash->end()) {
		return LIBRARY_MEMORY_ERROR;
	}

	vh = iter->second;

	ffmpegNextFrame(vh);

	err = videoHandle_toMImage(imgFuns, vh, &img);

	if (err == LIBRARY_NO_ERROR)
		MArgument_setMImage(Res, img);

	return err;
}

