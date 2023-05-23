#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include "fpioa.h"
#include "board_config.h"
#include "plic.h"
#include "sysctl.h"
#include "kpu.h"
#include <rtthread.h>
#define PLL0_OUTPUT_FREQ 800000000UL
#define PLL1_OUTPUT_FREQ 400000000UL

/****************************
 * please update this
 **************************/
#include "lcd.h"
#include "img0.h"
#include "img0_chw.h"
#include "rt_ai.h"
#include "logo_image.h"
#include "rt_ai_mnist_model.h"

#define MODEL_INPUT_W 28
#define MODEL_INPUT_H 28
#define LCD_WIDTH 320
#define LCD_HEIGHT 240

#define MY_MODEL_NAME           RT_AI_MNIST_MODEL_NAME
#define MY_MODEL_IN_1_SIZE      RT_AI_MNIST_IN_1_SIZE
#define MY_MODEL_OUT_1_SIZE     RT_AI_MNIST_OUT_1_SIZE

volatile rt_ai_uint32_t g_ai_done_flag;  // 判断模型推理一次是否完成
// {RGB565-IMG  , CHW-RGB-IMG}
uint8_t *datasets[][2] = {{IMG0, IMG0_CHW}, {IMG3, IMG3_CHW}, {IMG5, IMG5_CHW}, {IMG6, IMG6_CHW}, {IMG7, IMG7_CHW}, {IMG8, IMG8_CHW}, {IMG9, IMG9_CHW} };
char *label[] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
uint8_t kpu_img[28*28*1];
/********************************************************/
static void ai_done(void *ctx);

int main(void)
{
    /* Set CPU clock */
    sysctl_clock_enable(SYSCTL_CLOCK_AI);  // 使能系统时钟（系统时钟初始化）
    io_mux_init();  // 硬件引脚初始化
    io_set_power();  // 设置IO口电压

    /* LCD init */
    rt_kprintf("LCD init\n");
    lcd_init();
    
    /* LCD显示图片 */
    lcd_draw_picture(0, 0, LCD_WIDTH, LCD_HEIGHT, (rt_ai_uint16_t *)LOGO_IMAGE);
    lcd_draw_string(40, 40, "Hello RT-Thread!", RED);
    lcd_draw_string(40, 60, "Demo: Mnist", BLUE);
    sleep(1);

    rt_ai_t mymodel = NULL;  // 初始化模型信息


    /* AI modol inference */
    mymodel = rt_ai_find(MY_MODEL_NAME);  // 找到模型
    if(!mymodel)
    {
        rt_kprintf("\nmodel find error\n");
        while (1) {};
    }

    if (rt_ai_init(mymodel, (rt_ai_buffer_t *)kpu_img) != 0)  // 初始化模型，传入输入数据
    {
        rt_kprintf("\nmodel init error\n");
        while (1) {};
    }

    rt_kprintf("rt_ai_init complete..\n");
    int i = 0;
    while(1){
        i++;
        i %= 7;
        g_ai_done_flag = 0;
        memcpy(kpu_img, datasets[i][1], 28*28*1); //复制CHW IMG到kpu输入.
        if(rt_ai_run(mymodel, ai_done, NULL) != 0)
        {    // 模型推理一次
            rt_kprintf("rtak run fail!\n");
            while (1) {};
        }
        rt_kprintf("AI model inference done.\r\n");
        while(!g_ai_done_flag) {};  // 等待kpu运算完成

        float *output;  // 模型输出结果存放
        int prediction = -1;  // 模型预测结果
        float scores = 0.;  // 模型预测概率
        output = (float *)rt_ai_output(mymodel, 0);  // 获取模型输出结果
        /* 对模型输出结果进行处理，该实验是Mnist，输出结果为10个概率值，选出其中最大概率即可 */
        for(int i = 0; i < 10 ; i++)
        {
            // printf("pred: %d, scores: %.2f%%\n", i, output[i]*100);
            if(output[i] > scores  && output[i] > 0.2)
            {
                prediction = i;
                scores = output[i];
            }
        }
    
        /* 如果预测的概率值都没有超过0.2的，则视为预测失败*/
        if (scores == 0.)
        {
            rt_kprintf("no predictions here\n");
            lcd_clear(BLACK);   // 清屏
            while (1) {};
        }
        printf("The prediction: %d, scores: %.2f%%\n", prediction, scores*100);

        /* show result in LCD */
        lcd_clear(WHITE);   // 清屏
        lcd_draw_string(160, 100, label[prediction], RED);
        lcd_draw_picture(160, 120, 28, 28, (rt_ai_uint16_t *)datasets[i][0]);  // rgb565 图片显示
        sleep(2);
    }
    return 0;
}

static void ai_done(void *ctx)
{
    g_ai_done_flag = 1;
}
