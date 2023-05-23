#ifndef __RT_AI_MNIST_MODEL_H
#define __RT_AI_MNIST_MODEL_H

/* model info ... */

// model name
#define RT_AI_MNIST_MODEL_NAME			"mnist"

#define RT_AI_MNIST_WORK_BUFFER_BYTES	(16000)

#define AI_MNIST_DATA_WEIGHTS_SIZE		(1493120) //unused

#define RT_AI_MNIST_BUFFER_ALIGNMENT		(4)

#define RT_AI_MNIST_IN_NUM				(1)

#define RT_AI_MNIST_IN_1_SIZE			(1 * 1 * 28 * 28)
#define RT_AI_MNIST_IN_1_SIZE_BYTES		((1 * 1 * 28 * 28) * 4)
#define RT_AI_MNIST_IN_SIZE_BYTES		{	\
	((1 * 1 * 28 * 28) * 4) ,	\
}

#define RT_AI_MNIST_IN_TOTAL_SIZE_BYTES	((1 * 1 * 28 * 28) * 4)


#define RT_AI_MNIST_OUT_NUM				(1)

#define RT_AI_MNIST_OUT_1_SIZE			(1 * 10)
#define RT_AI_MNIST_OUT_1_SIZE_BYTES		((1 * 10) * 4)
#define RT_AI_MNIST_OUT_SIZE_BYTES		{	\
	((1 * 10) * 4) ,	\
}

#define RT_AI_MNIST_OUT_TOTAL_SIZE_BYTES	((1 * 10) * 4)



#define RT_AI_MNIST_TOTAL_BUFFER_SIZE		//unused

#endif	//end
