#include <rt_ai.h>
#include <backend_k210_kpu.h>
#include <rt_ai_mnist_model.h>
#include <kpu.h>

extern unsigned char mnist_kmodel[];

/* based on k210 */
#define RT_AI_MNIST_INFO    {       \
    RT_AI_MNIST_IN_NUM,             \
    RT_AI_MNIST_OUT_NUM,            \
    RT_AI_MNIST_IN_SIZE_BYTES,      \
    RT_AI_MNIST_OUT_SIZE_BYTES,     \
    RT_AI_MNIST_WORK_BUFFER_BYTES,  \
    ALLOC_INPUT_BUFFER_FLAG                 \
}

#define RT_AI_MNIST_HANDLE  {         \
    .info   =     RT_AI_MNIST_INFO    \
}

#define RT_K210_AI_MNIST   {   \
    .parent         = RT_AI_MNIST_HANDLE,   \
    .model          = mnist_kmodel, \
    .dmac           = DMAC_CHANNEL5,        \
}

static struct k210_kpu rt_k210_ai_mnist = RT_K210_AI_MNIST;

static int rt_k210_ai_mnist_init(){
    rt_ai_register(RT_AI_T(&rt_k210_ai_mnist),RT_AI_MNIST_MODEL_NAME,0,backend_k210_kpu,&rt_k210_ai_mnist);
    return 0;
}

INIT_APP_EXPORT(rt_k210_ai_mnist_init);
