/*
 * nvcc `pkg-config --cflags --libs gtk+-2.0` canny.cu -o canny
 */

#ifndef USING_CUDA_FUNCTION
#include    <cairo.h>
#include    <gtk/gtk.h>

#define DEBUG

#ifdef DEBUG
#include    <stdio.h>
#endif

#define IMAGE_NAME       "Resources/lena.png"
#endif /* USING_CUDA_FUNCTION */

#define WINDOW_WIDTH     512
#define WINDOW_HEIGHT    512
#define PADDING          10

#define BLOCKDIM         16

#define LOW_THRESHOLD    40
#define HIGH_THRESHOLD   60

#define CANDIDATE_PIXEL  0


__global__ void gaussianVert_kernel(mint * in, mint * out, mint width, mint height, mint channels, mint pitch) {
    __shared__ mint smem[BLOCKDIM][BLOCKDIM+6];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x,  by = blockIdx.y;
    int xIndex = tx + bx*BLOCKDIM;
    int yIndex = ty + by*BLOCKDIM;

    if (xIndex >= width)
        return;

    int index = channels*xIndex + yIndex*pitch;

    float accum;
    mint cc, * col = &in[xIndex*channels];

    for (cc = 0; cc < channels; cc++) {
        smem[tx][ty+3] = yIndex < height ? col[yIndex*pitch + cc] : 0;
        if (ty < 3) {
            smem[tx][ty] = (yIndex < height && yIndex > ty) ? col[(yIndex-3)*pitch + cc] : 0;
            smem[tx][ty+BLOCKDIM+3] = (yIndex+BLOCKDIM < height) ? col[(yIndex+BLOCKDIM)*pitch + cc] : 0;
        }
        __syncthreads();

        if (yIndex < height) {
            accum  = 0.006f * smem[tx][ty];
            accum += 0.061f * smem[tx][ty+1];
            accum += 0.242f * smem[tx][ty+2];
            accum += 0.383f * smem[tx][ty+3];
            accum += 0.242f * smem[tx][ty+4];
            accum += 0.061f * smem[tx][ty+5];
            accum += 0.006f * smem[tx][ty+6];
            out[index + cc] = accum;
        }
        __syncthreads();
    }
}

__global__ void gaussianHoriz_kernel(mint * in, mint * out, mint width, mint height, mint channels, mint pitch) {
    __shared__ mint smem[BLOCKDIM+2*3][BLOCKDIM];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x,  by = blockIdx.y;
    int xIndex = tx + bx*BLOCKDIM;
    int yIndex = ty + by*BLOCKDIM;

    if (yIndex >= height)
        return;

    int index = channels*xIndex + yIndex*pitch;

    float accum;
    mint cc, * row = &in[yIndex*pitch];

    for (cc = 0; cc < channels; cc++) {
        smem[tx+3][ty] = xIndex < width ? row[xIndex*channels + cc] : 0;
        if (tx < 3) {
            smem[tx][ty] = (xIndex < width && xIndex > tx) ? row[(xIndex-3)*channels + cc] : 0;
            smem[tx+BLOCKDIM+3][ty] = (xIndex+BLOCKDIM < width) ? row[(xIndex+BLOCKDIM)*channels + cc] : 0;
        }
        __syncthreads();

        if (xIndex < width) {
            accum  = 0.006f * smem[tx][ty];
            accum += 0.061f * smem[tx+1][ty];
            accum += 0.242f * smem[tx+2][ty];
            accum += 0.383f * smem[tx+3][ty];
            accum += 0.242f * smem[tx+4][ty];
            accum += 0.061f * smem[tx+5][ty];
            accum += 0.006f * smem[tx+6][ty];
            out[index + cc] = accum;
        }
        __syncthreads();
    }
}

__global__ void sobelGxVert_kernel(mint * in, mint * out, mint width, mint height, mint channels, mint pitch) {
    __shared__ mint smem[BLOCKDIM][BLOCKDIM+2];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x,  by = blockIdx.y;
    int xIndex = tx + bx*BLOCKDIM;
    int yIndex = ty + by*BLOCKDIM;

    if (xIndex >= width)
        return;

    int index = channels*xIndex + yIndex*pitch;

    mint cc, * col = &in[xIndex*channels];

    for (cc = 0; cc < channels; cc++) {
        smem[tx][ty+1] = (yIndex < height) ? col[yIndex*pitch + cc] : 0;
        if (ty == 0) {
            smem[tx][0] = (yIndex < height && yIndex > 0) ? col[(yIndex-1)*pitch + cc] : 0;
            smem[tx][BLOCKDIM+1] = (yIndex+BLOCKDIM < height) ? col[(yIndex+BLOCKDIM)*pitch + cc] : 0;
        }
        __syncthreads();

        if (yIndex < height)
            out[index + cc] = smem[tx][ty] + 2*smem[tx][ty+1] + smem[tx][ty+2];
        __syncthreads();
    }
}

__global__ void sobelGxHoriz_kernel(mint * in, mint * out, mint width, mint height, mint channels, mint pitch) {
    __shared__ mint smem[BLOCKDIM+2][BLOCKDIM];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x,  by = blockIdx.y;
    int xIndex = tx + bx*BLOCKDIM;
    int yIndex = ty + by*BLOCKDIM;

    if (yIndex >= height)
        return;

    int index = channels*xIndex + yIndex*pitch;

    mint cc, * row = &in[yIndex*pitch];

    for (cc = 0; cc < channels; cc++) {
        smem[tx+1][ty] = (xIndex < width) ? row[xIndex*channels + cc] : 0;
        if (tx == 0) {
            smem[0][ty] = (xIndex < width && xIndex > 0) ? row[(xIndex-1)*channels + cc] : 0;
            smem[BLOCKDIM+1][ty] = (xIndex+BLOCKDIM < width) ? row[(xIndex+BLOCKDIM)*channels + cc] : 0;
        }
        __syncthreads();

        if (xIndex < width)
            out[index + cc] = smem[tx][ty] - smem[tx+2][ty];
        __syncthreads();
    }
}

__global__ void sobelGyVert_kernel(mint * in, mint * out, mint width, mint height, mint channels, mint pitch) {
    __shared__ mint smem[BLOCKDIM][BLOCKDIM+2];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x,  by = blockIdx.y;
    int xIndex = tx + bx*BLOCKDIM;
    int yIndex = ty + by*BLOCKDIM;

    if (xIndex >= width)
        return;

    int index = channels*xIndex + yIndex*pitch;

    mint cc, * col = &in[xIndex*channels];

    for (cc = 0; cc < channels; cc++) {
        smem[tx][ty+1] = (yIndex < height) ? col[yIndex*pitch + cc] : 0;
        if (ty == 0) {
            smem[tx][0] = (yIndex < height && yIndex > 0) ? col[(yIndex-1)*pitch + cc] : 0;
            smem[tx][BLOCKDIM+1] = (yIndex+BLOCKDIM < height) ? col[(yIndex+BLOCKDIM)*pitch + cc] : 0;
        }
        __syncthreads();

        if (yIndex < height)
            out[index + cc] = smem[tx][ty] - smem[tx][ty+2];
        __syncthreads();
    }
}

__global__ void sobelGyHoriz_kernel(mint * in, mint * out, mint width, mint height, mint channels, mint pitch) {
    __shared__ mint smem[BLOCKDIM+2][BLOCKDIM];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x,  by = blockIdx.y;
    int xIndex = tx + bx*BLOCKDIM;
    int yIndex = ty + by*BLOCKDIM;

    if (yIndex >= height)
        return;

    int index = channels*xIndex + yIndex*pitch;

    mint cc, * row = &in[yIndex*pitch];

    for (cc = 0; cc < channels; cc++) {
        smem[tx+1][ty] = (xIndex < width) ? row[xIndex*channels + cc] : 0;
        if (tx == 0) {
            smem[0][ty] = (xIndex < width && xIndex > 0) ? row[(xIndex-1)*channels + cc] : 0;
            smem[BLOCKDIM+1][ty] = (xIndex+BLOCKDIM < width) ? row[(xIndex+BLOCKDIM)*channels + cc] : 0;
        }
        __syncthreads();

        if (xIndex < width)
            out[index + cc] = smem[tx][ty] + 2*smem[tx+1][ty] + smem[tx+2][ty];
        __syncthreads();
    }
}

__global__ void magnitude_kernel(mint * gx, mint * gy, mint * out, mint width, mint height, mint channels, mint pitch) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x,  by = blockIdx.y;
    int xIndex = tx + bx*BLOCKDIM;
    int yIndex = ty + by*BLOCKDIM;

    int index = channels*xIndex + yIndex*pitch;
    mint cc;

    if (yIndex < height) {
        for (cc = 0; cc < channels; cc++)
            out[index + cc] = hypotf(gx[index+cc], gy[index+cc]);
    }
}

__global__ void atan_kernel(mint * gx, mint * gy, float * out, mint width, mint height, mint channels, mint pitch) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x,  by = blockIdx.y;
    int xIndex = tx + bx*BLOCKDIM;
    int yIndex = ty + by*BLOCKDIM;

    int index = channels*xIndex + yIndex*pitch;
    mint cc, d;

    if (xIndex < width && yIndex < height) {
        for (cc = 0; cc < channels; cc++) {
            d = gx[index+cc];    
            if (d != 0)
                out[index + cc] = atanf((float) gy[index+cc]/d);
            else
                out[index + cc] = gy[index+cc] ? 90.0 : 0.0;
        }
    }
}

__global__ void zeroCrossing_kernel(float * in, mint * out, mint width, mint height, mint channels, mint pitch) {
    mint tx = threadIdx.x, ty = threadIdx.y;
    mint bx = blockIdx.x,  by = blockIdx.y;
    mint xIndex = tx + bx*BLOCKDIM;
    mint yIndex = ty + by*BLOCKDIM;

    mint index = channels*xIndex + yIndex*pitch;
    mint cc;
    float d;

    if (xIndex < width && yIndex < height) {
        for (cc = 0; cc < channels; cc++) {
            d = in[index+cc];    
            if (d <= 22.5f || d >= 157.5f) {
                out[index+cc] = 0;
            } else if (d <= 67.5f) {
                out[index+cc] = 45;
            } else if (d <= 112.5f) {
                out[index+cc] = 90;
            } else {
                out[index+cc] = 135;
            }
        }
    }
}

__global__ void nonMaximalSuppression_kernel(mint * in, mint * direction, mint * out, mint width, mint height, mint channels, mint pitch) {
    __shared__ mint smem[BLOCKDIM+2][BLOCKDIM+2];    

    int tx = threadIdx.x,   ty = threadIdx.y;
    int bx = blockIdx.x,    by = blockIdx.y;
    int xIndex = tx + bx*BLOCKDIM;
    int yIndex = ty + by*BLOCKDIM;
    int index = xIndex*channels + yIndex*pitch;
    mint ii, jj, cc, d;

    for (cc = 0, ii = tx, jj = ty; cc < channels; cc++) {
        smem[tx+1][ty+1] = (xIndex < width && yIndex < height) ? in[index + cc] : 0;
        if (tx == 0) {
            smem[0][ty+1] = (xIndex > 0 && yIndex < height) ? in[index + cc - channels] : 0;
            smem[BLOCKDIM+1][ty+1] = (xIndex+BLOCKDIM < width && yIndex < height) ? in[index + cc + channels*BLOCKDIM] : 0;
        }
        if (ty == 0) {
            smem[tx+1][0] = (xIndex < width && yIndex > ty) ? in[index + cc - pitch] : 0;
            smem[tx+1][BLOCKDIM+1] = (yIndex+BLOCKDIM < height && xIndex < width) ? in[index + cc + BLOCKDIM*pitch] : 0;
        }

        if (tx == 0 && ty == 0) {
            if (xIndex > 0 && yIndex > 0)
                smem[0][0] = (xIndex < width && yIndex < width) ? in[index + cc - BLOCKDIM*pitch - channels] : 0;
            else
                smem[0][0] = 0;

            if (xIndex > 0 && yIndex+BLOCKDIM < height)
                smem[0][BLOCKDIM+1] = (xIndex < width) ? in[index + cc - channels + BLOCKDIM*pitch] : 0;
            else
                smem[0][BLOCKDIM+1] = 0;

            if (xIndex+BLOCKDIM < width && yIndex > 0)
                smem[BLOCKDIM+1][0] = (yIndex < height) ? in[index + cc + BLOCKDIM*channels - pitch] : 0;
            else
                smem[BLOCKDIM+1][0] = 0;

            if (xIndex+BLOCKDIM < width && yIndex+BLOCKDIM < height)
                smem[BLOCKDIM+1][BLOCKDIM+1] = in[index + cc + channels + BLOCKDIM*pitch];
            else
                smem[BLOCKDIM+1][BLOCKDIM+1] = 0;
        }
        __syncthreads();
        
        tx++;
        ty++;

        if (xIndex < width && yIndex < height) {
            d = direction[index+cc];
            if (d == 0) {
                if (smem[tx][ty+1] >= smem[tx][ty] || smem[tx][ty-1] >= smem[tx][ty])
                    out[index+cc] = 0;
                else
                    out[index+cc] = smem[tx][ty];
            } else if (d == 45) {
                if (smem[tx-1][ty-1] >= smem[tx][ty] || smem[tx+1][ty+1] >= smem[tx][ty])
                    out[index+cc] = 0;
                else
                    out[index+cc] = smem[tx][ty];
            } else if (d == 90) {
                if (smem[tx-1][ty] >= smem[tx][ty] || smem[tx+1][ty] >= smem[tx][ty])
                    out[index+cc] = 0;
                else
                    out[index+cc] = smem[tx][ty];
            } else {
                if (smem[tx+1][ty-1] >= smem[tx][ty] || smem[tx-1][ty+1] >= smem[tx][ty])
                    out[index+cc] = 0;
                else
                    out[index+cc] = smem[tx][ty];
            }
        }
        tx = ii;
        ty = jj;
        __syncthreads();
    }
}

__global__ void hysteresis_kernel(mint * in, mint * out, mint width, mint height, mint channels, mint pitch) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x,  by = blockIdx.y;
    int xIndex = tx + bx*BLOCKDIM;
    int yIndex = ty + by*BLOCKDIM;

    int index = channels*xIndex + yIndex*pitch;
    mint cc, px;

    if (xIndex < width && yIndex < height) {
        for (cc = 0; cc < channels; cc++) {
            px = in[index+cc];
            if (px >= HIGH_THRESHOLD)
                out[index + cc] = 255;
            else if (px <= LOW_THRESHOLD)
                out[index + cc] = 0;
            else
                out[index + cc] = CANDIDATE_PIXEL; // should do another pass.
        }
    }
}

__global__ void binarize_kernel(mint * in, mint * out, mint width, mint height, mint channels, mint pitch) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x,  by = blockIdx.y;
    int xIndex = tx + bx*BLOCKDIM;
    int yIndex = ty + by*BLOCKDIM;

    int index = channels*xIndex + yIndex*pitch;
    mint cc, accum;

    if (xIndex < width && yIndex < height) {
        for (cc = 0, accum = 0; cc < channels; cc++)
            accum += in[index+cc];
        if (accum) {
            for (cc = 0; cc < channels; cc++)
                out[index+cc] = 255;
        } else {
            for (cc = 0; cc < channels; cc++)
                out[index+cc] = 0;
        }
    }
}

void cannyEdgeDetection(mint * in, mint * out,
                        mint width, mint height, mint channels, mint pitch) {
    mint * d_in, * d_out, * d_tmp, * d_gx, * d_gy, * d_c, * d_mag;
    float * d_tan;

    cudaMalloc((void **) &d_in, height*pitch*sizeof(int));
    cudaMalloc((void **) &d_out, height*pitch*sizeof(int));
    cudaMalloc((void **) &d_tmp, height*pitch*sizeof(int));
    cudaMalloc((void **) &d_gx, height*pitch*sizeof(int));
    cudaMalloc((void **) &d_gy, height*pitch*sizeof(int));
    cudaMalloc((void **) &d_c, height*pitch*sizeof(int));
    cudaMalloc((void **) &d_mag, height*pitch*sizeof(int));
    cudaMalloc((void **) &d_tan, height*pitch*sizeof(float));

    cudaMemcpy(d_in, in, height*pitch*sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCKDIM, BLOCKDIM);
    dim3 gridDim((width + blockDim.x - 1)/blockDim.x, (height + blockDim.y - 1)/blockDim.y);

    gaussianHoriz_kernel<<<gridDim, blockDim>>>(d_in, d_tmp, width, height, channels, pitch);
    cudaThreadSynchronize();
    gaussianVert_kernel<<<gridDim, blockDim>>>(d_tmp, d_out, width, height, channels, pitch);
    cudaThreadSynchronize();

    sobelGxHoriz_kernel<<<gridDim, blockDim>>>(d_out, d_tmp, width, height, channels, pitch);
    cudaThreadSynchronize();
    sobelGxVert_kernel<<<gridDim, blockDim>>>(d_tmp, d_gx, width, height, channels, pitch);
    cudaThreadSynchronize();

    sobelGyHoriz_kernel<<<gridDim, blockDim>>>(d_out, d_tmp, width, height, channels, pitch);
    cudaThreadSynchronize();
    sobelGyVert_kernel<<<gridDim, blockDim>>>(d_tmp, d_gy, width, height, channels, pitch);
    cudaThreadSynchronize();

    magnitude_kernel<<<gridDim, blockDim>>>(d_gx, d_gy, d_mag, width, height, channels, pitch);
    cudaThreadSynchronize();

    atan_kernel<<<gridDim, blockDim>>>(d_gx, d_gy, d_tan, width, height, channels, pitch);
    cudaThreadSynchronize();

    zeroCrossing_kernel<<<gridDim, blockDim>>>(d_tan, d_c, width, height, channels, pitch);
    cudaThreadSynchronize();

    nonMaximalSuppression_kernel<<<gridDim, blockDim>>>(d_mag, d_c, d_tmp, width, height, channels, pitch);
    cudaThreadSynchronize();

    hysteresis_kernel<<<gridDim, blockDim>>>(d_tmp, d_out, width, height, channels, pitch);
    cudaThreadSynchronize();

    binarize_kernel<<<gridDim, blockDim>>>(d_out, d_out, width, height, channels, pitch);
    cudaThreadSynchronize();

    cudaMemcpy(out, d_out, height*pitch*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_tmp);
    cudaFree(d_gx);
    cudaFree(d_gy);
    cudaFree(d_c);
    cudaFree(d_tan);
    cudaFree(d_mag);
}

#ifndef USING_CUDA_FUNCTION
void uchar2int(int n, unsigned char * in, mint * out) {
    mint ii;

    for (ii = 0; ii < n; ii++) {
        if (ii/4)
            out[ii] = (int) in[ii];
        else
            out[ii] = 0;
    }
}

void int2uchar(int n, mint * in, unsigned char * out) {
    mint ii;

    for (ii = 0; ii < n; ii++) {
        if (ii/4)
            out[ii] = (unsigned char) in[ii] & 0xFF;
        else
            out[ii] = 0;
    }
}

int on_expose_event(GtkWidget *widget, GdkEventExpose *event, cairo_surface_t * img) {
    cairo_surface_t * new_img;
    unsigned char * pixels;
    mint * ipixels, * opixels;
    mint imageWidth, imageHeight, imagePitch, imageChannels;
    cairo_t * cx;
    cairo_format_t imageFormat;
    
    cx = gdk_cairo_create (widget->window);

    imageFormat = cairo_image_surface_get_format(img);
    switch (imageFormat) { // assuming the format is RGB24
        case CAIRO_FORMAT_ARGB32:
            printf("ARGB32 format not supported\n");
            exit(1);
        case CAIRO_FORMAT_A8:
            printf("A8 format not supported\n");
            exit(1);
        case CAIRO_FORMAT_A1:
            printf("A1 format not supported\n");
            exit(1);
    }
    imageChannels = 4;  //RGB24 format

    imageWidth = cairo_image_surface_get_width(img);
    imageHeight = cairo_image_surface_get_height(img);
    imagePitch = cairo_image_surface_get_stride(img);

    ipixels = (mint *) malloc(imageHeight*imagePitch*sizeof(int));
    opixels = (mint *) malloc(imageHeight*imagePitch*sizeof(int));

    pixels = cairo_image_surface_get_data(img);

    uchar2int(imageHeight*imagePitch, pixels, ipixels);
    
    cannyEdgeDetection(ipixels, opixels, imageWidth, imageHeight, imageChannels, imagePitch);

    int2uchar(imageHeight*imagePitch, opixels, pixels);

    new_img = cairo_image_surface_create_for_data(pixels, imageFormat,
                                                  imageWidth, imageHeight,
                                                  imagePitch);
    
    cairo_scale(cx, (1.0-0.01*PADDING)*WINDOW_WIDTH/imageWidth,
                    (1.0-0.01*PADDING)*WINDOW_HEIGHT/imageHeight);
    cairo_translate(cx, (WINDOW_WIDTH - (1.0-0.01*PADDING)*WINDOW_WIDTH)/2.0,
                        (WINDOW_HEIGHT - (1.0-0.01*PADDING)*WINDOW_HEIGHT)/2.0);
    cairo_set_source_surface(cx, new_img, 0, 0);

    cairo_paint(cx);
    
    cairo_destroy(cx);

    printf("performed canny edge detection\n");
    
    return FALSE;
}

int main(int argc, char ** argv) {
    GtkWidget * window;
    cairo_surface_t * img;
    
    img = cairo_image_surface_create_from_png(IMAGE_NAME);
    
    gtk_init(&argc, &argv);
    
    window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    
    g_signal_connect(window, "expose-event", G_CALLBACK (on_expose_event), img);
    g_signal_connect(window, "destroy", G_CALLBACK (gtk_main_quit), NULL);
    
    gtk_window_set_position(GTK_WINDOW(window), GTK_WIN_POS_CENTER);
    gtk_window_set_default_size(GTK_WINDOW(window), WINDOW_WIDTH, WINDOW_HEIGHT); 
    gtk_widget_set_app_paintable(window, TRUE);
    
    gtk_widget_show_all(window);
    
    gtk_main();
    
    cairo_surface_destroy(img);
    
    return 0;
}
#endif /* USING_CUDA_FUNCTION */
