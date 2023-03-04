
#include<cuda_runtime.h>

#define min(a, b)  ((a) < (b) ? (a) : (b))
#define num_threads   512
typedef unsigned char uint8_t;

struct Size{
    int width = 0, height = 0;

    Size() = default;
    Size(int w, int h)
    :width(w), height(h){}
};

struct AffineMatrix{
    float i2d[6]; float d2i[6];
    void invertAffineTransform(float imat[6], float omat[6]){
        float i00 = imat[0];  float i01 = imat[1];  float i02 = imat[2];
        float i10 = imat[3];  float i11 = imat[4];  float i12 = imat[5];

        // 计算行列式
        float D = i00 * i11 - i01 * i10;
        D = D != 0 ? 1.0 / D : 0;

        // 计算剩余的伴随矩阵除以行列式
        float A11 = i11 * D;
        float A22 = i00 * D;
        float A12 = -i01 * D;
        float A21 = -i10 * D;
        float b1 = -A11 * i02 - A12 * i12;
        float b2 = -A21 * i02 - A22 * i12;
        omat[0] = A11;  omat[1] = A12;  omat[2] = b1;
        omat[3] = A21;  omat[4] = A22;  omat[5] = b2;
    }
    void compute(const Size& from , const Size& to){
    float scale_x = to.width / (float)from.width;

    float scale_y = to.height / (float)from.height;
    float scale = min(scale_x, scale_y);

    i2d[0] = scale;  i2d[1] = 0;  i2d[2] = (-scale * from.width  + to.width + scale  - 1) * 0.5;
    i2d[3] = 0;  i2d[4] = scale;  i2d[5] = (-scale * from.height + to.height + scale - 1) * 0.5;
    invertAffineTransform(i2d, d2i);

    }

};
   
__device__ void affine_project(float* matrix, int x, int y, float* proj_x, float* proj_y){
    *proj_x = matrix[0] *x + matrix[1] *y + matrix[2] +0.5f;
    *proj_y = matrix[3] *x + matrix[4] *y + matrix[5] +0.5f;
}
   
__global__ void warp_affine_bilinear_kernel(uint8_t* src, int src_line_size,int src_width, int src_height,
                                            uint8_t* dst,int dst_line_size, int dst_width, int dst_height,
                                            uint8_t fill_value, AffineMatrix matrix){ 

    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if(dx>=dst_width || dy >= dst_height) return;
    float src_x =0;float src_y = 0;

 

    affine_project(matrix.d2i,dx,dy,&src_x,&src_y);


    // float c0 = 114;float c1=114;float c2=114;
    uint8_t  c0 = 114 ,c1=114,c2=114;


    if(src_x <-1 || src_x >= src_width || src_y<=-1 || src_y>= src_height){
    }else{
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);  
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uint8_t const_values[] = {fill_value,fill_value,fill_value};
        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;
        float w1 = hy*hx;
        float w2= hy*lx;
        float w3= ly*hx;
        float w4= ly*lx;

        uint8_t* v1 = const_values;
        uint8_t* v2 = const_values;
        uint8_t* v3 = const_values;
        uint8_t* v4 = const_values;

        if(y_low>=0){
            if(x_low >= 0)
            v1 = src + y_low*src_line_size + x_low*3;
            if(x_high <src_width)
            v2 = src + y_low*src_line_size + x_high*3;
            
        }
        if(y_high < src_height){
            if(x_low >= 0)
            v3 = src + y_high*src_line_size + x_low*3;
            if(x_high <src_width)
            v4 = src + y_high*src_line_size + x_high*3;  
                  }
        // c0 = min(floorf(w1*v1[0] + w2*v2[0] + w3*v3[0]+w4*v4[0] +0.5f),255);
        // c1 = min(floorf(w1*v1[1] + w2*v2[1] + w3*v3[1]+w4*v4[1] +0.5f),255);
        // c2 = min(floorf(w1*v1[2] + w2*v2[2] + w3*v3[2]+w4*v4[2] +0.5f),255);
        c0 = floorf(w1*v1[0] + w2*v2[0] + w3*v3[0]+w4*v4[0] +0.5f);
        c1 = floorf(w1*v1[1] + w2*v2[1] + w3*v3[1]+w4*v4[1] +0.5f);
        c2 = floorf(w1*v1[2] + w2*v2[2] + w3*v3[2]+w4*v4[2] +0.5f);

    }

    // #把计算好的值，放到目标图像对应的位置
    uint8_t* pdst = dst + dy*dst_line_size + dx*3;
    pdst[0]=  c0;
    pdst[1]=  c1;
    pdst[2]=  c2;
}

void warp_affine_bilinear(
       uint8_t* src, int src_line_size, int src_width, int src_height, 
    uint8_t* dst, int dst_line_size, int dst_width, int dst_height, 
	uint8_t fill_value){
    dim3 block_size(32,32);
    dim3 grid_size((dst_width + 31)/32,(dst_height +31)/32);
    AffineMatrix affine;
    affine.compute(Size(src_width,src_height),Size(dst_width,dst_height));

    warp_affine_bilinear_kernel<<< grid_size,block_size,0,nullptr>>>(        
        src, src_line_size, src_width, src_height,
        dst, dst_line_size, dst_width, dst_height,
        fill_value,affine);

}



__global__ void warp_affine_bilinear_kernel1(uint8_t* src, int src_line_size,int src_width, int src_height,
                                            uint8_t* dst,int dst_line_size, int dst_width, int dst_height,
                                            uint8_t fill_value, AffineMatrix matrix){ 

    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if(dx>=dst_width || dy >= dst_height) return;
    float src_x =0;float src_y = 0;

 

    affine_project(matrix.d2i,dx,dy,&src_x,&src_y);


    // float c0 = 114;float c1=114;float c2=114;
    uint8_t  c0 = 114 ;


    if(src_x <-1 || src_x >= src_width || src_y<=-1 || src_y>= src_height){
    }else{
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);  
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uint8_t const_values= fill_value;
        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;
        float w1 = hy*hx;
        float w2= hy*lx;
        float w3= ly*hx;
        float w4= ly*lx;

        uint8_t* v1 = &const_values;
        uint8_t* v2 = &const_values;
        uint8_t* v3 = &const_values;
        uint8_t* v4 = &const_values;

        if(y_low>=0){
            if(x_low >= 0)
            v1 = src + y_low*src_line_size + x_low;
            if(x_high <src_width)
            v2 = src + y_low*src_line_size + x_high;
            
        }
        if(y_high < src_height){
            if(x_low >= 0)
            v3 = src + y_high*src_line_size + x_low;
            if(x_high <src_width)
            v4 = src + y_high*src_line_size + x_high;  
                  }

        c0 = floorf(w1*v1[0] + w2*v2[0] + w3*v3[0]+w4*v4[0] +0.5f);


    }

    // #把计算好的值，放到目标图像对应的位置
    uint8_t* pdst = dst + dy*dst_line_size + dx;
    pdst[0]=  c0;
    }

void warp_affine_bilinear1(
       uint8_t* src, int src_line_size, int src_width, int src_height, 
    uint8_t* dst, int dst_line_size, int dst_width, int dst_height, 
	uint8_t fill_value){
    dim3 block_size(32,32);
    dim3 grid_size((dst_width + 31)/32,(dst_height +31)/32);
    AffineMatrix affine;
    affine.compute(Size(src_width,src_height),Size(dst_width,dst_height));

    warp_affine_bilinear_kernel1<<< grid_size,block_size,0,nullptr>>>(        
        src, src_line_size, src_width, src_height,
        dst, dst_line_size, dst_width, dst_height,
        fill_value,affine);

}
