/*SketchTemplate.cpp*/

/*file that is uploaded to the Arduino Uno*/

/*Author: Ian Johnson, Wolfram Research Inc.*/

/*Copyright (c) 2015 All rights reserved.*/

/*
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*COMPILED ON: `time`*/

/*SYSTEM INFO: `sysInfo`*/





#ifndef LONG2DARRAY_H
#define	LONG2DARRAY_H

#include "Arduino.h"

/*GENERATED LIBRARY INCLUDES FOLLOW*/

`libraries`

/*END GENERATED LIBRARY INCLUDES*/


/*GENERATED USER DEFINED INTITIALIZATIONS FOLLOWS*/

`initializations`

/*END GENERATED USER DEFINED INTITIALIZATIONS*/



/*there values are taken from the firmata protocol, available here: http://wolfr.am/2HCysYhr (case sensitive)*/
#define DIGITAL_READ_TASK 0xD0
#define DIGITAL_WRITE_TASK 0x90
#define ANALOG_WRITE_TASK 0xE0
#define ANALOG_READ_TASK 0xC0
#define COMMAND_TASK 0xF0
#define CONFIG_TASK 0xF4
#define SYSEX_START 0xF0
#define SYSEX_END 0xF7
#define STRING 0x71


/*TODO: combine these two lists of functions so they are the same task types*/

/*these are custom firmata commands that will get sent over the wire*/
#define OUT_OF_MEMORY 0x00
#define FUNCTION_CALL_DELETE 0x01
#define FUNCTION_CALL_ADD 0x02
#define FLUSH_SERIAL 0x03
#define FLOAT_NUM 0x04
#define LONG_NUM 0x05
#define FLOAT_ARRAY 0x06
#define LONG_ARRAY 0x07

/*this is for the task types from the task struct for running a task*/
#define CONFIG 0
#define DIGITAL_READ 1
#define DIGITAL_WRITE 2
#define ANALOG_READ 3
#define ANALOG_WRITE 4
#define FUNCTION_CALL 5
#define FUNCTION_DELETE 6


#define ADC_TO_VOLTAGE_CONV 0.00488759



#define LENGTH(x) sizeof(x)/sizeof(x[0])
#define getBitValue(someByte, bitNum) ((1<<(bitNum)) & (someByte))>>(bitNum)

/*magical AVR constants to determine if a pointer is on the heap or not (i.e. if it needs to be free'd)*/
#define isPointerDynamic(c) ((int)(c) < AVR_STACK_POINTER_REG && (int)(c) > (int)__malloc_heap_start)


/*from the avr libraries, for clearing (cbi) and setting (sbi) bits in bytes*/
#ifndef cbi
#define cbi(sfr, bit) (_SFR_BYTE(sfr) &= ~_BV(bit))
#endif
#ifndef sbi
#define sbi(sfr, bit) (_SFR_BYTE(sfr) |= _BV(bit))
#endif 

#define DEBUG_PRINT 0

#ifndef AVR_BAUD_RATE
#define AVR_BAUD_RATE 115200
#endif

/*for the arduino uno, just use the normal serial connection*/
#if defined (ARDUINO_AVR_UNO)
#define SerialLink Serial

//this is the same on the arduino yun when we use it normally as it's connected to the computer
#elif defined (ARDUINO_AVR_YUN)
#define SerialLink Serial
#endif

//on the yun, this is the baud rate that we use to talk to the Atheros Linux processor
#define ATHEROS_BAUD_RATE 250000

//for common cloud definitions - so we only use one definition for both cloud api implementations
#if defined(ARDUINO_AVR_YUN) && (defined(YUN_DATADROP_FUNCTIONS) || defined(YUN_CHANNELFRAMEWORK_FUNCTIONS))

//for storing / accessing variables in program memory space
#include <avr/pgmspace.h>

//the curl command is shared between both the channel framework and data drop, so we don't want to 
//duplicate that memory usage, even if it is just like 8 bytes. EVERY BYTE COUNTS!!
const PROGMEM char curlCommand[] = {"curl -k "};
/****************************************************************/
//NOTE TO DEVELOPER : CHANGE THIS DEFINE IF THE URL EVER CHANGES!!!
/****************************************************************/
#define CURL_COMMAND_LENGTH 8

//if we store these URLs in program memory, the api requests are slower as we have to read out the stirng each time
//but it saves us on RAM, as then the values can just live in one single place in RAM, and only when we use it
#if defined(YUN_DATADROP_FUNCTIONS)
const PROGMEM char databinURL[] = {"https://datadrop.wolframcloud.com/api/v1.0/Add?bin="};
//because this is constant, we can define it's length and save on having to determine it's length dynamically
/****************************************************************/
//NOTE TO DEVELOPER : CHANGE THIS DEFINE IF THE URL EVER CHANGES!!!
/****************************************************************/
#define DATABIN_URL_LENGTH 51 
#endif

#if defined(YUN_CHANNELFRAMEWORK_FUNCTIONS)
const PROGMEM char channelFrameworkURL[] = {"https://channelbroker.wolframcloud.com/"};
//same deal for the channel url as the databin
/****************************************************************/
//NOTE TO DEVELOPER : CHANGE THIS DEFINE IF THE URL EVER CHANGES!!!
/****************************************************************/
#define CHANNEL_URL_LENGTH 39
#endif

//we provide differently named and typed so that the functions can be used in C code
//but internally the different value types get casted to a String anyways to get called via the Atheros,
//so we can just use one internal implementation
void cloudSend(char * cloudIdPath, char ** keyNames, String * values, size_t len, const char * apiURL)
{
    //wait til the atheros processor has had time to start up in case this is called right when 
    //we get power
    while(millis() < 60000);

    //get the string length of the channel framework base URL
    int apiStrLen;
#if defined(YUN_CHANNELFRAMEWORK_FUNCTIONS)
    if(apiURL == channelFrameworkURL)
    {
        apiStrLen = CHANNEL_URL_LENGTH;
    }
#endif
    //databin case
#if defined(YUN_DATADROP_FUNCTIONS)
#if defined(YUN_CHANNELFRAMEWORK_FUNCTIONS)
    else
#endif
    if(apiURL == databinURL)
    {
        apiStrLen = DATABIN_URL_LENGTH;
    }
#endif

    int curlStrLen = CURL_COMMAND_LENGTH;
    
    //print off a newline to clear out any possible commands lingering in the terminal on the Atheros
    Serial1.println();

    //because we are storing the string in program memory, we can't just read it normally, so
    //iterate over it getting each character and writing it out to the command line
    int index;
    for (index = 0; index < curlStrLen - 1; index++)
    {
        Serial1.write(pgm_read_byte_near(curlCommand + index));
    }

    //write a space to break up the curl command and the url to call
    Serial1.write(' ');

    //now write all the constant characters from the api url
    for (index = 0; index < apiStrLen - 1; index++)
    {
        Serial1.write(pgm_read_byte_near(apiURL + index));
    }

    //check which api we are using
#if defined(YUN_CHANNELFRAMEWORK_FUNCTIONS)
    if(apiURL == channelFrameworkURL)
    {
        //add the channel path without any qualifiers
        Serial1.print(cloudIdPath);
        
        //now we have to add the set mode parameter
        //note we have to escape the ampersand, else the bash prompt thinks we are trying to run the 
        //previous string as a full command in the background
        Serial1.print("?operation=send");
    }
#endif
    //databin case
#if defined(YUN_DATADROP_FUNCTIONS)
#if defined(YUN_CHANNELFRAMEWORK_FUNCTIONS)
    else
#endif
    if(apiURL == databinURL)
    {
       //now print the cloudIdPath, which in this case is the databin ID
        Serial1.print(cloudIdPath);
    }
#endif

    //now loop over the keys and values, adding them as we go
    for(index = 0; index < len; index++)
    {
        //first print an ampersand to split up the parameters
        //note we have to escape the ampersand because the ash shell in OpenWRT tries to fork all of the previous
        //characters as a previous command
        Serial1.write('\\');
        Serial1.write('&');
        
        //print off the name of this parameter
        Serial1.print(keyNames[index]);

        //print off the equal sign next
        Serial1.write('=');

        //finally print off the value for this key
        Serial1.print(values[index]);
    }
    
    //finaly print off a newline to run the command and clear out the command line
    Serial1.println();
}


#endif

//for datadrop functionality these convenience functions are provided
#if defined(ARDUINO_AVR_YUN) && defined(YUN_DATADROP_FUNCTIONS)

//because the verbatim code is the exact same for all 3 types, but the compiler needs to see it in different contexts
//we save ourselves from duplicating the code base with this simple define
#define databinSendCode() \
do \
{ \
    String stringVal(val); \ 
    cloudSend(binID,&keyName,&stringVal,1,databinURL); \
} while(0)

//same sort of define for array sendings
#define databinSendArrayCode() \
do \
{ \
    String * stringVals = (String *) calloc(len,sizeof(String)); \
    int index; \
    for(index = 0; index < len; index++) \
    { \
        stringVals[index] = String(vals[index]); \
    } \
    cloudSend(binID,keyNames,stringVals,len,databinURL); \
    free(stringVals); \
} while(0)

//for integers
void DatabinIntegerAdd(char * binID,char * keyName, long val)
{
   databinSendCode();
}

//for reals (doubles) - note on most AVR platforms double is the same as a float
void DatabinRealAdd(char * binID, char * keyName, double val)
{
    databinSendCode();
}

//for strings
void DatabinStringAdd(char * binID, char * keyName, char * val)
{
    databinSendCode();
}

//for reals (doubles) - note on most AVR platforms double is the same as a float
void DatabinRealArrayAdd(char * binID, char ** keyNames, double * vals, size_t len)
{
    databinSendArrayCode();
}

void DatabinIntegerArrayAdd(char * binID, char ** keyNames, long * vals, size_t len)
{
    databinSendArrayCode();
}

#endif


//for channel framework publishing functionality
#if defined(ARDUINO_AVR_YUN) && defined(YUN_CHANNELFRAMEWORK_FUNCTIONS)

//because the verbatim code is the exact same for all 3 types, but the compiler needs to see it in different contexts
//we save ourselves from duplicating the code base with this simple define
#define channelSendCode() \
do \
{ \
    String stringVal(val); \ 
    cloudSend(channelPath,&keyName,&stringVal,1,channelFrameworkURL); \
} while(0)

//same sort of define for array sendings
#define channelArraySendCode() \
do \
{ \
    String * stringVals = (String *) calloc(len,sizeof(String)); \
    int index; \
    for(index = 0; index < len; index++) \
    { \
        stringVals[index] = String(vals[index]); \
    } \
    cloudSend(channelPath,keyNames,stringVals,len,channelFrameworkURL); \
    free(stringVals); \
} while(0)

//for integers
void ChannelIntegerSend(char * channelPath,char * keyName, long val)
{
    channelSendCode();
}

//for reals (doubles) - note on most AVR platforms double is the same as a float
void ChannelRealSend(char * channelPath, char * keyName, double val)
{
    channelSendCode();
}

//for strings
void ChannelStringSend(char * channelPath, char * keyName, char * val)
{
    channelSendCode();
}

void ChannelIntegerArraySend(char * channelPath,char ** keyNames, long * vals, size_t len)
{
    channelArraySendCode();
}

void ChannelRealArraySend(char * channelPath,char ** keyNames, double * vals, size_t len)
{
    channelArraySendCode();
}


#endif


/*GENERATED USER FUNCTION DEFINITIONS FOLLOW*/

`userFunctionSources`

/*END GENERATED USER FUNCTION DEFINITIONS*/


extern "C" {



typedef struct
{
    byte numArrays;
    byte * arrayLengths;
    long ** theArrays;
} long2DArray;

typedef struct
{
    byte numArrays;
    byte * arrayLengths;
    float ** theArrays;
} float2DArray;

typedef struct
{
    byte numArrays;
    byte * arrayLengths;
    char ** theArrays;
} char2DArray;

typedef struct arguments
{
    /*for each of these, the high nibble of the byte is the number 
    for the first type and the low nibble is the number for the 
    second type, i.e. for numLongAndFloatArgs, the number of long arguments
    is contained in the high nibble of numLongAndFloatArgs, while the number
    of float arguments is the low nibble of numLongAndFloatArgs*/
    byte numLongAndFloatArgs;
    byte numStringAndArrayArgs;
    byte numLongArrayAnd2DLongArrayArgs;
    byte numFloatArrayAnd2DFloatArrayArgs;
    byte numStringArrayAnd2DStringArrayArgs;
    /*the commented out members are unimplemented due to memory space concerns on the arduino*/
    long * longArgs;
    long2DArray * longArrayArgs;
    /*long * long2DArrayArgs;*/
    float * floatArgs;
    float2DArray * floatArrayArgs;
    /* float * float2DArrayArgs;*/
    char2DArray * stringArgs;
    /* char * stringArrayArgs;*/
    /* char * string2DArrayArgs;*/
} arguments;


    
typedef struct
{
    /*id is used to figure out what specific function should be called*/
    byte id;
    /*type is the type of the task, 0 is config, 1 is write, etc.*/
    byte type;
    /*argsSignature is a string where each char is a type of argument*/
    char * argSignature;
    /*taskArgs stores all of the arguments information*/
    arguments * taskArgs;
    /*waitTime is the amount of time in millis remaining before*/
    /*the task should be run, this will be decremented as time marches on*/
    unsigned long waitTime;
    /*numTimesToRun is the number of iterations*/
    unsigned long numTimesToRun;
    /*syncTime is the amount of time to wait between calls*/
    unsigned long syncTime;
    /*timeToRun is the amount of time that should be spent running the*/
    /*task*/
    unsigned long timeToRun;
    /*lastTimeUpdated stores the last time that the task's timer was updated*/
    unsigned long lastTimeUpdated;
    /*interruptableInfinite stores whether or not the task is infinite, */
    /*and whether or not the task is interruptable*/
    byte interruptableInfinite;
    byte portNumber;
    byte pinNumber;
    int value;
} task;


typedef struct node
{
    /*pointer for next node in the chain*/
    struct node * next;
    /*pointer for the previous node in the chain*/
    struct node * previous;
    /*the task at this node*/
    task * data;
} node;


typedef struct queue
{
    /*the size will make it convinient for us in several functions*/
    byte size;
    /*a pointer to the first node in the queue (aka the front node)*/
    node * first;
    /*a pointer to he last node in the queue (aka the back node)*/
    node * last;
} queue;


union Converter
{
  unsigned long rawBytes;
  long theLong;
  float theFloat;
};






/**
	setNumArraysLong - sets the number of arrays in this 2d array
*/
void setNumArraysLong( long2DArray * matrix, byte numberArrays)
{
    byte dataArrayPointerIndex;
    /*firstly, we need to check if the pointer is not null, if it is*/
    /*then we need to free the memory previously referenced there*/
    if (matrix->arrayLengths)
    {
        free(matrix->arrayLengths);
        /*now the arrayLengths matrix is good to be allocated, but we still*/
        /*need to check the actual array that stores the data pointers*/
        if (matrix->theArrays)
        {
            /*this means that we previously allocated memory for storing the data,*/
            /*so before we free the array of pointers, we need to iterate*/
            /*through each pointer, freeing them all*/

            for (dataArrayPointerIndex = 0; dataArrayPointerIndex < matrix->numArrays; dataArrayPointerIndex++)
            {
                /*need to free each pointer*/
                free((matrix->theArrays)[dataArrayPointerIndex]);
            }
            /*now all the individual data arrays are cleared, so now we can */
            /*free the array of pointers to the data arrays itself*/
			free(matrix->theArrays);
        }
    }
    /*now the memory has been safely freed, and we can allocate new memory if necessary*/
	/*if the number of arrays is zero, don't do anything, it's redundant, and furthermore*/
	/*malloc will actually allocate one byte with malloc(0)*/
    /*first set the number of arrays*/
    matrix->numArrays = numberArrays;
	if(numberArrays)
	{
		/*then allocate memory for the length of each array*/
		matrix->arrayLengths = (byte *)calloc(numberArrays,sizeof(byte));
		/*then allocate memory for the pointers to the arrays themselves*/
		matrix->theArrays = (long **)calloc(numberArrays,sizeof(long *));
	}
}


/**
    setGivenArrayLengthLong - sets the length of the nth array in the 2d array
*/
inline void setGivenArrayLengthLong( long2DArray * matrix, byte arrayNum, byte arrayLength)
{
    (matrix->arrayLengths)[arrayNum] = arrayLength;
}


/**
	setGivenArrayLong - sets the nth array to be what was passed
*/
void setGivenArrayLong( long2DArray * matrix, byte arrayNum, byte arrayLength, long * theArray)
{
    if ((matrix->theArrays)[arrayNum])
    {
        /*the pointer is not null, so we have to free that memory before we create new*/
        /*memory with malloc*/
        free((matrix->theArrays)[arrayNum]);
    }
    /*the pointer is now null, so we can safely make new memory for it*/
    /*first though, check to make sure that arrayLengths was initialized so*/
    /*that it's safe for us to reference the array of pointers*/
    if(matrix->arrayLengths)
    {
		/*first check to make sure arrayLength isn't zero, if it is we don't have to do anything*/
		if(arrayLength)
		{
			/*first malloc some memory for the array*/
			(matrix->theArrays)[arrayNum] = (long *) calloc(arrayLength,sizeof(long));
			/*then copy the array to the pointer at the position in the full array*/
			memcpy((matrix->theArrays)[arrayNum],theArray,arrayLength*sizeof(long));
		}
		/*finally, set the length, regardless of if it is zero or not*/
        setGivenArrayLengthLong(matrix,arrayNum,arrayLength);
    }
}


/**
    getNumArraysLong - gets the number of arrays in this 2d array
*/
inline byte getNumArraysLong( long2DArray * matrix)
{
    return matrix->numArrays;
}


/**
    getGivenArrayLengthLong - gets the length of the nth array
*/
inline byte getGivenArrayLengthLong( long2DArray * matrix, byte arrayNum)
{
    return (matrix->arrayLengths)[arrayNum];
}


/**
    getGivenArrayLong - gets the nth array
*/
long * getGivenArrayLong( long2DArray * matrix, byte arrayNum)
{
	/*save this as a local variable to prevent having to call the function multiple times*/
	byte givenArrayLength = getGivenArrayLengthLong(matrix,arrayNum);
    /*first allocate new memory so that the returned pointer isn't pointing at the internal array*/
    long * returnPointer;
    returnPointer = (long *) calloc(givenArrayLength,sizeof(float));
    /*check if the memory was properly allocated*/
    if (returnPointer)
    {
        /*we have the memory, so copy over the internal array to this new pointer*/
        memcpy(returnPointer,(matrix->theArrays)[arrayNum],givenArrayLength*sizeof(long));
        return returnPointer;
    }
    else
    {
		/*out of memory, return null pointer*/
		return 0;
    }
	
}


/**
    safeDeleteLong will safely free all of the memory associated with the given 2d long array
*/
void safeDeleteLong( long2DArray * matrix)
{
	/*first check to make sure matrix exists*/
	if(matrix)
	{
		/*to safely free all the memory associated with the 2d array, we have to */
		/*free the following all the pointers stored in the struct, including all*/
		/*the pointers in the theArrays array*/
		byte arrayIndex;
		/*first free all the individual arrays*/
		byte totalArrayNum = getNumArraysLong(matrix);
		for(arrayIndex = 0; arrayIndex < totalArrayNum; arrayIndex++)
		{
			free((matrix->theArrays)[arrayIndex]);
		}
		/*next we can free the pointer to those pointers*/
		free(matrix->theArrays);
		/*next we can free the array of array lengths*/
		free(matrix->arrayLengths);
		/*if the pointer is on the heap, free it as well*/
		if(isPointerDynamic(matrix))
		{
			free(matrix);
		}
	}
}




#endif	/* LONG2DARRAY_H */


/**
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
*/


/* 
 * File:   float2darray.h
 * Author: Ian
 *
 * Created on January 12, 2015, 7:24 PM
 */

#ifndef FLOAT2DARRAY_H
#define	FLOAT2DARRAY_H





/**
    setNumArraysFloat - sets the number of arrays in this 2d array
*/
void setNumArraysFloat( float2DArray * matrix, byte numberArrays)
{
    byte dataArrayPointerIndex;
    /*firstly, we need to check if the pointer is not null, if it is*/
    /*then we need to free the memory previously referenced there*/
    if (matrix->arrayLengths)
    {
        free(matrix->arrayLengths);
        /*now the arrayLengths matrix is good to be allocated, but we still*/
        /*need to check the actual array that stores the data pointers*/
        if (matrix->theArrays)
        {
            /*this means that we previously allocated memory for storing the data,*/
            /*so before we free the array of pointers, we need to iterate*/
            /*through each pointer, freeing them all*/

            for (dataArrayPointerIndex = 0; dataArrayPointerIndex < matrix->numArrays; dataArrayPointerIndex++)
            {
                /*need to free each pointer*/
                free((matrix->theArrays)[dataArrayPointerIndex]);
            }
            /*now all the individual data arrays are cleared, so now we can */
            /*free the array of pointers to the data arrays itself*/
			free(matrix->theArrays);
        }
    }
    /*now we are good to allocate memory for the data*/
	/*first set the number of arrays*/
    matrix->numArrays = numberArrays;
	/*next, check to see if the number of arrays is not zero, if it is zero, nothing needs to be*/
	/*done*/
    if(numberArrays)
	{
		/*then allocate memory for the length of each array*/
		matrix->arrayLengths = (byte *)calloc(numberArrays,sizeof(byte));
		/*then allocate memory for the pointers to the arrays themselves*/
		matrix->theArrays = (float **) calloc(numberArrays,sizeof(float *));
	}
}


/**
    setGivenArrayLengthFloat - sets the length of the nth array in the 2d array
*/
inline void setGivenArrayLengthFloat( float2DArray * matrix, byte arrayNum, byte arrayLength)
{
    (matrix->arrayLengths)[arrayNum] = arrayLength;
}


/**
    setGivenArrayFloat - sets the nth array to be what was passed
*/
void setGivenArrayFloat( float2DArray * matrix, byte arrayNum, byte arrayLength, float * theArray)
{
    if ((matrix->theArrays)[arrayNum])
    {
        /*the pointer is not null, so we have to free that memory before we create new*/
        /*memory with malloc*/
        free((matrix->theArrays)[arrayNum]);
    }
    /*the pointer is now null, so we can safely make new memory for it*/
    /*first though, check to make sure that arrayLengths was initialized so*/
    /*that it's safe for us to reference the array of pointers*/
    if(matrix->arrayLengths)
    {
		/*only actually need to allocate memory if arrayLength isn't 0*/
		if(arrayLength)
		{
			/*first malloc some memory for the array*/
			(matrix->theArrays)[arrayNum] = (float *) calloc(arrayLength,sizeof(float));
			/*then copy the array to the pointer at the position in the full array*/
			memcpy((matrix->theArrays)[arrayNum],theArray,arrayLength*sizeof(float));
		}
		/*regardless set the length of this array*/
        setGivenArrayLengthFloat(matrix,arrayNum,arrayLength);
    }
}


/**
    getNumArraysFloat - gets the number of arrays in this 2d array
*/
inline byte getNumArraysFloat( float2DArray * matrix)
{
    return matrix->numArrays;
}


/**
 *  getGivenArrayLengthFloat - gets the length of the nth array of this 2d array
 *
 *  Note that whether or not this includes the null terminator is dependent on
 *  whether or not the user included it when they passed in the length with setGivenArrayFloat 
 *  or setGivenArrayLengthFloat.
 *  It is recommended that the user include the null terminator in this, else
 *  when using returning the pointer to this string, it may not be recognized as
 *  a string properly by other c functions like printf("%s"), etc.
 *  If the user used strlen, then this will not include the null terminator. so
 *  the user should add 1 to the result of strlen
*/
inline byte getGivenArrayLengthFloat( float2DArray * matrix, byte arrayNum)
{
    return (matrix->arrayLengths)[arrayNum];
}


/**
    getGivenArrayFloat - gets the nth array out of this 2d array
*/
float * getGivenArrayFloat( float2DArray * matrix, byte arrayNum)
{
	/*save this as a local variable to prevent having to call the function multiple times*/
	byte givenArrayLength = getGivenArrayLengthFloat(matrix,arrayNum);
    /*first allocate new memory so that the returned pointer isn't pointing at the internal array*/
    float * returnPointer;
    returnPointer = (float *) calloc(givenArrayLength,sizeof(float));
    /*check if the memory was properly allocated*/
    if (returnPointer)
    {
        /*we have memory, so copy over the internal array to this new pointer*/
        memcpy(returnPointer,(matrix->theArrays)[arrayNum],givenArrayLength*sizeof(float));
        return returnPointer;
    }
    else
    {
		/*no more memory available*/
        return 0;
    }
}


/**
    safeDeleteFloat will safely free all of the memory associated with the given 2d floatacter array
*/
void safeDeleteFloat( float2DArray * matrix)
{
	/*check to make sure that matrix isn't a null pointer*/
	if(matrix)
	{
		/*to safely free all the memory associated with the 2d array, we have to */
		/*free the following all the pointers stored in the struct, including all*/
		/*the pointers in the theArrays array*/
		byte arrayIndex;
		/*first free all the individual arrays*/
		byte totalArrayNum = getNumArraysFloat(matrix);
		for(arrayIndex = 0; arrayIndex < totalArrayNum; arrayIndex++)
		{
			free(matrix->theArrays[arrayIndex]);
		}
		/*next we can free the pointer to those pointers*/
		free(matrix->theArrays);
		/*next we can free the array of array lengths*/
		free(matrix->arrayLengths);
		/*if the pointer is on the heap, free it and null it as well*/
		if(isPointerDynamic(matrix))
		{
			free(matrix);
		}
	}
}



#endif /* FLOAT2DARRAY_H */

/**
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
*/


/* 
 * File:   char2darray.h
 * Author: Ian
 *
 * Created on January 12, 2015, 7:24 PM
 */

#ifndef CHAR2DARRAY_H
#define	CHAR2DARRAY_H





/**
    setNumArraysChar - sets the number of arrays in this 2d array
*/
void setNumArraysChar( char2DArray * matrix, byte numberArrays)
{
    byte dataArrayPointerIndex;
    /*firstly, we need to check if the pointer is not null, if it is*/
    /*then we need to free the memory previously referenced there*/
    if (matrix->arrayLengths)
    {
		
        free(matrix->arrayLengths);
		
        /*now the arrayLengths matrix is good to be allocated, but we still*/
        /*need to check the actual array that stores the data pointers*/
        if (matrix->theArrays)
        {
            /*this means that we previously allocated memory for storing the data,*/
            /*so before we free the array of pointers, we need to iterate*/
            /*through each pointer, freeing them all*/
            for (dataArrayPointerIndex = 0; dataArrayPointerIndex < matrix->numArrays; dataArrayPointerIndex++)
            {
                /*need to free each pointer*/
                free((matrix->theArrays)[dataArrayPointerIndex]);
            }
			
            /*now all the individual data arrays are cleared, so now we can */
            /*free the array of pointers to the data arrays itself*/
			free(matrix->theArrays);
        }
    }
    /*now we are good to allocate memory for the data*/
	/*only actually allocate memory if numberArrays isn't zero though*/
	if(numberArrays)
	{
		/*then allocate memory for the length of each array*/
		matrix->arrayLengths = (byte *)calloc(numberArrays,sizeof(byte));
		/*then allocate memory for the pointers to the arrays themselves*/
		matrix->theArrays = (char **) calloc(numberArrays,sizeof(char *));
	}
	/*finally, regardless of the number of arrays, set the number of arrays*/
    matrix->numArrays = numberArrays;
}


/**
    setGivenArrayLengthChar - sets the length of the nth array in the 2d array
*/
inline void setGivenArrayLengthChar( char2DArray * matrix, byte arrayNum, byte arrayLength)
{
    (matrix->arrayLengths)[arrayNum] = arrayLength;
}


/**
    setGivenArrayChar - sets the nth array to be what was passed
*/
void setGivenArrayChar( char2DArray * matrix, byte arrayNum, byte arrayLength, char * theArray)
{
    if ((matrix->theArrays)[arrayNum])
    {
        /*the pointer is not null, so we have to free that memory before we create new*/
        /*memory with malloc*/
        free((matrix->theArrays)[arrayNum]);
    }
    /*the pointer is now null, so we can safely make new memory for it*/
    /*first though, check to make sure that arrayLengths was initialized so*/
    /*that it's safe for us to reference the array of pointers*/
    if(matrix->arrayLengths != 0)
    {
		/*check to see if arrayLength is zero, if it is then we don't need to copy anything*/
		if(arrayLength)
		{
			/*first malloc some memory for the array*/
			(matrix->theArrays)[arrayNum] = (char *) calloc(arrayLength,sizeof(char));
			/*then copy the array to the pointer at the position in the full array*/
			memcpy((matrix->theArrays)[arrayNum],theArray,arrayLength*sizeof(char));
		}
		/*regardless set the length still*/
        setGivenArrayLengthChar(matrix,arrayNum,arrayLength);
    }
}


/**
    getNumArraysChar - gets the number of arrays in this 2d array
*/
inline byte getNumArraysChar( char2DArray * matrix)
{
    return matrix->numArrays;
}


/**
    getGivenArrayLengthChar - gets the length of the nth array of this 2d array

    Note that whether or not this includes the null terminator is dependent on
    whether or not the user included it when they passed in the length with setGivenArrayChar 
    or setGivenArrayLengthChar.
 *  It is recommended that the user include the null terminator in this, else
 *  when using returning the pointer to this string, it may not be recognized as
 *  a string properly by other c functions like printf("%s"), etc.
    If the user used strlen, then this will not include the null terminator. so
 *  the user should add 1 to the result of strlen
*/
inline byte getGivenArrayLengthChar( char2DArray * matrix, byte arrayNum)
{
    return (matrix->arrayLengths)[arrayNum];
}


/**
    getGivenArrayChar - gets the nth array out of this 2d array
*/
char * getGivenArrayChar( char2DArray * matrix, byte arrayNum)
{
	byte givenArrayLength = getGivenArrayLengthChar(matrix,arrayNum);
    /*first allocate new memory so that the returned pointer isn't pointing at the internal array*/
    char * returnPointer;
    returnPointer = (char *) calloc(givenArrayLength,sizeof(char));
    /*check if the memory was properly allocated*/
    if (returnPointer)
    {
        /*we have the memory, so copy over the internal array to this new pointer*/
        memcpy(returnPointer,(matrix->theArrays)[arrayNum],givenArrayLength*sizeof(char));
        return returnPointer;
    }
    else
    {
		/*no more memory available*/
        return 0;
    }
	
}


/**
    safeDeleteChar will safely free all of the memory associated with the given 2d character array
*/
void safeDeleteChar( char2DArray * matrix)
{
	/*make sure the matrix exists before we free it*/
	if(matrix)
	{
		/*to safely free all the memory associated with the 2d array, we have to */
		/*free the following all the pointers stored in the struct, including all*/
		/*the pointers in the theArrays array*/
		byte arrayIndex;
		/*first free all the individual arrays*/
		byte totalArrayNum = getNumArraysChar(matrix);
		for(arrayIndex = 0; arrayIndex < totalArrayNum; arrayIndex++)
		{
			free(matrix->theArrays[arrayIndex]);
		}
		/*next we can free the pointer to those pointers*/
		free(matrix->theArrays);
		/*next we can free the array of array lengths*/
		free(matrix->arrayLengths);
		/*if the pointer is on the heap, then free it and null it*/
		if(isPointerDynamic(matrix))
		{
			free(matrix);
		}
	}
}





#endif	/* CHAR2DARRAY_H */



/**
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
*/


/* 
 * File:   arguments.h
 * Author: Ian
 *
 * Created on January 12, 2015, 7:27 PM
 */



#ifndef ARGUMENTS_H
#define	ARGUMENTS_H




/*===================================================================*/
/*===================================================================*/
/*=================ARGUMENTS ADT=====================================*/
/*===================================================================*/
/*===================================================================*/
/*===================================================================*/







/**
    getLongArgNum - gets the number of long arguments

    To get the number of long args, we take the high nibble of the 
    numLongAndFloatArgs member of the struct
*/
inline byte getLongArgNum( arguments * args )
{
    /*bitshift numLongAndFloatArgs 4 to the right to get the high nibble*/
    return (args->numLongAndFloatArgs) >> 4;
}


/**
    getFloatArgNum - gets the number of float arguments

    To get the number of float args, we take the low nibble of the 
    numLongAndFloatArgs member of the struct
*/
inline byte getFloatArgNum( arguments * args )
{
    /*bitwise and the numLongAndFloatArgs with 0b00001111 to get the low nibble*/
    return (args->numLongAndFloatArgs) & 15;
}


/**
    getStringArgNum - gets the number of string arguments

    To get the number of string args, we take the high nibble of the 
    numStringAndArrayArgs member of the struct
*/
inline byte getStringArgNum( arguments * args)
{
    /*bitshift numStringAndArrayArgs 4 to the right to get the high nibble*/
    return (args->numStringAndArrayArgs) >> 4;
}


/**
    getArrayArgNum - gets the number of array arguments (of either 1 or 2 dimensions)

    To get the number of array arguments, we take the low nibble of the numStringAndArrayArgs
    member of the struct
*/
inline byte getArrayArgNum( arguments * args)
{
    /*bitwise and numStringAndArrayArgs with 0b00001111 (15) to get the low nibble*/
    return (args->numStringAndArrayArgs) & 15;
}



/**
    getLongArrayArgNum - gets the number of long array arguments

    To get the number of long array args, we take the high nibble of the 
    numLongArrayAnd2DLongArrayArgs member of the struct
*/
inline byte getLongArrayArgNum( arguments * args)
{
    /*to get the high nibble, we bit shift numLongArrayAnd2DLongArrayArgs to right 4*/
    return (args->numLongArrayAnd2DLongArrayArgs) >> 4;
}


/**
    getFloatArrayArgNum - sets the number of float array arguments

    To get the number of float array args, we take the high nibble of the 
    numLongArrayAnd2DLongArrayArgs member of the struct
*/
inline byte getFloatArrayArgNum( arguments * args)
{
    /*to get the high nibble, we bit shift numFloatArrayAnd2DFloatArrayArgs to the right 4*/
    return (args->numFloatArrayAnd2DFloatArrayArgs) >> 4;
}


/** getArgNum - returns the total number of arguments
 * 
 * @param args - the argument adt to work on
 * @return the total number of arguments stored in this adt
 */
inline long getArgNum(arguments * args)
{
    return getLongArgNum(args)+
			getFloatArgNum(args)+
			getStringArgNum(args)+
            getArrayArgNum(args);
}


/**
    setLongArgNum - sets the number of long arguments

    To set the number of long arguments, we have to set the high nibble of the 
    numLongAndFloatArgs member of the struct to be what the user passed

    @param numLongs - the number of longs
*/
inline void setLongArgNum( arguments * args, byte numLongs)
{
    /*to set the high nibble of numLongAndFloatArgs, we bitshift numLongs to the left 4, and 
    bitwise or that with the bitwise and of numLongAndFloatArgs and 0b00001111*/
    args->numLongAndFloatArgs &= 15;
	args->numLongAndFloatArgs |= (numLongs << 4);
}

/**
    setFloatArgNum - sets the number of float arguments

    To set the number of float arguments, we have to set the low nibble of the numLongAndFloatArgs 
    member of the struct to be what the user passed

    @param numFloats - the number of floats
*/
inline void setFloatArgNum( arguments * args, byte numFloats)
{
    /*to set the low nibble of numLongAndFloatArgs, we have to bitwise and numLongAndFloatArgs
    with 0b11110000, then bitwise or that with numFloats bitwise anded with 0b00001111*/
    args->numLongAndFloatArgs &= 240;
	args->numLongAndFloatArgs |= (numFloats & 15);
}


/**
    setStringArgNum - sets the number of string arguments

    To set the number of string arguments, we have to set the high nibble of the 
    numStringArrayAnd2DStringArrayArgs member of the struct to be what the user passed

    @param numStrings - the number of strings
*/
inline void setStringArgNum( arguments * args, byte numStrings)
{
    /*to set the high nibble of numStringArrayAnd2DStringArrayArgs, we have to bitwise and it 
    with 0b00001111 to clear the high nibble, then bitwise or numStringArrayAnd2DStringArrayArgs
    with numStrings bitshifted to the left 4*/
    args->numStringAndArrayArgs &= 15;
	args->numStringAndArrayArgs |= (numStrings << 4);
}


/**
    setLongArrayArgNum - sets the number of long array arguments

    To set the number of long array arguments, we have to set the high nibble of 
    the numLongArrayAnd2DLongArrayArgs member of the struct to be what the user passed

    @param numLongArrays - the number of long arrays
*/
inline void setLongArrayArgNum( arguments * args, byte numLongArrays)
{
    /*to set the high nibble of numLongArrayAnd2DLongArrayArgs, we first have to clear the high
    nibble by bitwise anding it with 0b00001111, then we can bitshift numLongArrays to the
    left 4, then bitwise or that with numLongArrayAnd2DLongArrayArgs*/
    args->numLongArrayAnd2DLongArrayArgs &= 15;
	args->numLongArrayAnd2DLongArrayArgs |= (numLongArrays << 4 );
    /*we also have to increment the low nibble of numStringAndArrayArgs by whatever the user
    passed*/
	byte newArrayArgNum = (((args->numStringAndArrayArgs & 15) + numLongArrays) & 15);
    args->numStringAndArrayArgs &= 240;
	args->numStringAndArrayArgs |= newArrayArgNum;
}
	
	
/**
    setFloatArrayArgNum - sets the number of float array arguments

    To set the number of float array arguments, we have to set the high nibble of 
    the numFloatArrayAnd2DFloatArrayArgs member of the struct to be what the user passes in

    @param numFloatArrays - the number of float arrays
*/
inline void setFloatArrayArgNum( arguments * args, byte numFloatArrays)
{
    /*to set the high nibble of numFloatArrayAnd2DFloatArrayArgs, we first clear the high nibble
    by bitwise anding numFloatArrayAnd2DFloatArrayArgs with 0b00001111, then bitshifting
    numFloatArrays to the left 4, then bitwise oring the two together*/
    args->numFloatArrayAnd2DFloatArrayArgs &= 15;
	args->numFloatArrayAnd2DFloatArrayArgs |= (numFloatArrays << 4 );
    /*we also have to increment the low nibble of numStringAndArrayArgs by whatever the user
    passed*/
	byte newArrayArgNum = (((args->numStringAndArrayArgs & 15) + numFloatArrays) & 15);
    args->numStringAndArrayArgs &= 240;
	args->numStringAndArrayArgs |= newArrayArgNum;
                                                                    
}



/**
    getLongArgArray - gets an array of long arguments
*/
long * getLongArgArray( arguments * args)
{
	/*save this as a local variable to prevent having to call the function multiple times*/
	byte arrayLength = getLongArgNum(args);
    /*we allocate new memory for the return pointer, so that the returned pointer isn't pointing at
    the internal array*/
    long * returnPointer = 0;
	/*check if we even need to allocate anything to return*/
	if(getLongArgNum(args))
	{
		/*allocate new memory for the returned argument*/
		returnPointer= (long *)calloc(arrayLength,sizeof(long));
		if (returnPointer)
		{
			/*then copy the array over*/
			memcpy(returnPointer,args->longArgs,arrayLength*sizeof(long));
		}
	}
	/*because we initialized it to null, if there aren't any args or if memory allocation failed, we can still return the pointer*/
	return returnPointer;
    
}


/**
    getFloatArgArray - gets an array of float arguments
*/
float * getFloatArgArray( arguments * args)
{
	/*save this as a local variable to prevent having to call the function multiple times*/
	byte arrayLength = getFloatArgNum(args);
    /*we allocate new memory for the return pointer, so that the returned pointer isn't pointing at
    the internal array*/
    float * returnPointer = 0;
	/*check if we need to allocate anything to return*/
	if(getFloatArgNum(args))
	{
		/*allocate new memory for the returned argument*/
		returnPointer= (float *)calloc(arrayLength,sizeof(float));
		if(returnPointer)
		{
			/*then copy the array over and return it*/
			memcpy(returnPointer,args->floatArgs,arrayLength*sizeof(float));
		}
	}
    return returnPointer;
}


/**
    getStringArgArray - gets an array of string arguments
*/
char2DArray * getStringArgArray( arguments * args)
{
	char2DArray * returnPointer = 0;
	/*first check to make sure it exists in the struct*/
	if(args->stringArgs)
	{
		/*save this value in a local variable to prevent having to call the function multiple times*/
		byte totalArrayNum = getNumArraysChar(args->stringArgs);
		/*make a copy of the internal struct, and pass that back*/
		returnPointer = (char2DArray *) calloc(1,sizeof(char2DArray));
		/*first set the number of arrays of the returnPointer 2d array struct*/
		/*because we know that the matrix was just allocated*/
		/*we can optimize this to not have to call the setNumArraysChar function*/
		returnPointer->numArrays = totalArrayNum;
		returnPointer->arrayLengths = (byte *)calloc(totalArrayNum,sizeof(byte));
		returnPointer->theArrays = (char **)calloc(totalArrayNum,sizeof(char *));
		char * arrayToSet;
		byte arrayIndex;
		/*for each string in the internal struct, add that to the internal 2d array */
		/*struct*/
		char * currentArray;
		byte currentArrayLength;
		for (arrayIndex = 0; arrayIndex < totalArrayNum; arrayIndex++)
		{
			/*first get the array*/
			currentArray = ((args->stringArgs)->theArrays)[arrayIndex];
			/*now its length*/
			currentArrayLength = ((args->stringArgs)->arrayLengths)[arrayIndex];
			/*now make a new array for the destination*/
			arrayToSet = (char *)calloc(currentArrayLength,sizeof(char));
			/*copy over the memory from the currentArray to the destination*/
			memcpy(arrayToSet,currentArray,currentArrayLength*sizeof(char));
			/*now set the length of this array*/
			(returnPointer->arrayLengths)[arrayIndex] = currentArrayLength;
			/*finally point the returnPointer to this*/
			(returnPointer->theArrays)[arrayIndex] = arrayToSet;
		}
	}
	return returnPointer;
}


/**
    getLongArrayArgArray - gets an array of arrays of longs arguments
*/
long2DArray * getLongArrayArgArray( arguments * args)
{
	/*check to make sure that the argument set has long array arguments*/
    long2DArray * returnPointer = 0;
	/*first check to make sure it exists in the struct*/
	if(args->longArrayArgs)
	{
		/*save this value in a local variable to prevent having to call the function multiple times*/
		byte totalArrayNum = getNumArraysLong(args->longArrayArgs);
		/*make a copy of the internal struct, and pass that back*/
		returnPointer = (long2DArray *) calloc(1,sizeof(long2DArray));
		/*first set the number of arrays of the returnPointer 2d array struct*/
		/*because we know that the matrix was just allocated*/
		/*we can optimize this to not have to call the setNumArraysChar function*/
		returnPointer->numArrays = totalArrayNum;
		returnPointer->arrayLengths = (byte *)calloc(totalArrayNum,sizeof(byte));
		returnPointer->theArrays = (long **)calloc(totalArrayNum,sizeof(long *));
		long * arrayToSet;
		byte arrayIndex;
		/*for each string in the internal struct, add that to the internal 2d array */
		/*struct*/
		long * currentArray;
		byte currentArrayLength;
		for (arrayIndex = 0; arrayIndex < totalArrayNum; arrayIndex++)
		{
			/*first get the array*/
			currentArray = ((args->longArrayArgs)->theArrays)[arrayIndex];
			/*now its length*/
			currentArrayLength = ((args->longArrayArgs)->arrayLengths)[arrayIndex];
			/*now make a new array for the destination*/
			arrayToSet = (long *)calloc(currentArrayLength,sizeof(long));
			/*copy over the memory from the currentArray to the destination*/
			memcpy(arrayToSet,currentArray,currentArrayLength*sizeof(long));
			/*now set the length of this array*/
			(returnPointer->arrayLengths)[arrayIndex] = currentArrayLength;
			/*finally point the returnPointer to this*/
			(returnPointer->theArrays)[arrayIndex] = arrayToSet;
		}
	}
	return returnPointer;
}


/**
    getFloatArrayArgArray - gets an array of arrays of floats arguments
*/
float2DArray * getFloatArrayArgArray( arguments * args)
{
    	float2DArray * returnPointer = 0;
	/*first check to make sure it exists in the struct*/
	if(args->floatArrayArgs)
	{
		/*save this value in a local variable to prevent having to call the function multiple times*/
		byte totalArrayNum = getNumArraysFloat(args->floatArrayArgs);
		/*make a copy of the internal struct, and pass that back*/
		returnPointer = (float2DArray *) calloc(1,sizeof(float2DArray));
		/*first set the number of arrays of the returnPointer 2d array struct*/
		/*because we know that the matrix was just allocated*/
		/*we can optimize this to not have to call the setNumArraysChar function*/
		returnPointer->numArrays = totalArrayNum;
		returnPointer->arrayLengths = (byte *)calloc(totalArrayNum,sizeof(byte));
		returnPointer->theArrays = (float **)calloc(totalArrayNum,sizeof(float *));
		float * arrayToSet;
		byte arrayIndex;
		/*for each string in the internal struct, add that to the internal 2d array */
		/*struct*/
		float * currentArray;
		byte currentArrayLength;
		for (arrayIndex = 0; arrayIndex < totalArrayNum; arrayIndex++)
		{
			/*first get the array*/
			currentArray = ((args->floatArrayArgs)->theArrays)[arrayIndex];
			/*now its length*/
			currentArrayLength = ((args->floatArrayArgs)->arrayLengths)[arrayIndex];
			/*now make a new array for the destination*/
			arrayToSet = (float *)calloc(currentArrayLength,sizeof(float));
			/*copy over the memory from the currentArray to the destination*/
			memcpy(arrayToSet,currentArray,currentArrayLength*sizeof(float));
			/*now set the length of this array*/
			(returnPointer->arrayLengths)[arrayIndex] = currentArrayLength;
			/*finally point the returnPointer to this*/
			(returnPointer->theArrays)[arrayIndex] = arrayToSet;
		}
	}
	return returnPointer;
}



 
/**
    setLongArgArray - sets an array of long arguments

    @param longArray - the array of long arguments
    @param lengthOfArray - the length of the array that is passed
*/
void setLongArgArray( arguments * args, long * longArray, byte lengthOfArray)
{
    if (args->longArgs)
    {
        /*the pointer isn't null, so we have to free the memory,*/
        free(args->longArgs);
    }
	/*check to see if longArray exists*/
	if(longArray)
	{
		/*allocate some memory for the array*/
		args->longArgs = (long *)calloc(lengthOfArray,sizeof(long));
		/*copy the array over to the pointer in the struct*/
		if (args->longArgs)
		{
			/*make sure that it allocated properly*/
			memcpy(args->longArgs, longArray, lengthOfArray*sizeof(long));
		}
	}
	else
	{
		/*the user passed in a null pointer, so set the pointer to null*/
		args->longArgs = 0;
	}
	/*lastly, set the size of this array*/
	setLongArgNum(args,lengthOfArray);
}


/**
    setFloatArgArray - sets an array of float arguments

    @param floatArray - the array of float arguments
    @param lengthOfArray - the length of the array that is passed
*/
void setFloatArgArray( arguments * args, float * floatArray, byte lengthOfArray)
{
    /*first we need to check if the internal pointer is null, if it's not,*/
    /*we have to free that memory before we allocate new memory*/
    if (args->floatArgs)
    {
        /*it's not null, so free that memory*/
        free(args->floatArgs);
    }
	/*check to see if the float array passed in exists*/
	if(floatArray)
	{
		/*then allocate some memory for the array*/
		args->floatArgs = (float *) calloc(lengthOfArray,sizeof(float));
		/*copy the array over to the pointer in the struct*/
		if (args->floatArgs)
		{
			/*allocation successful*/
			memcpy(args->floatArgs, floatArray, lengthOfArray*sizeof(float));
		}
	}
	else
	{
		/*the user passed in a null */
		args->floatArgs = 0;
	}
	/*lastly, set the size of this array*/
	setFloatArgNum(args,lengthOfArray);
}


/**
    setStringArgArray - sets an array of string arguments

    @param stringArray - the 2darray ADT containing the string array arguments
*/
void setStringArgArray( arguments * args, char2DArray * stringArray)
{
	if(args->stringArgs)
	{
		/*then it was previously allocated, so we need to free it*/
		safeDeleteChar(args->stringArgs);
	}
	/*now check to see if stringArray exists*/
	if(stringArray)
	{
		/*save this as a local variable so we don't have to call the function multiple times*/
		byte totalArrayNum = getNumArraysChar(stringArray);
		/*now we know it is free, we can allocate a new one*/
		args->stringArgs = (char2DArray *) calloc(1,sizeof(char2DArray));
		/*because we know that the matrix was just allocated*/
		/*we can optimize this to not have to call the setNumArraysChar function*/
		(args->stringArgs)->numArrays = totalArrayNum;
		(args->stringArgs)->arrayLengths = (byte *)calloc(totalArrayNum,sizeof(byte));
		(args->stringArgs)->theArrays = (char **)calloc(totalArrayNum,sizeof(char *));
		/*also update the number of args structure with this number*/
		setStringArgNum(args,totalArrayNum);
		char * arrayToSet;
		byte arrayIndex;
		char * currentArray;
		byte currentArrayLength;
		/*for each string in the passed struct, add that to the internal 2d array struct*/
		for (arrayIndex = 0; arrayIndex < totalArrayNum; arrayIndex++)
		{
			/*first get the array*/
			currentArray = (stringArray->theArrays)[arrayIndex];
			/*now its length*/
			currentArrayLength = (stringArray->arrayLengths)[arrayIndex];
			/*now make a new array for the destination*/
			arrayToSet = (char *)calloc(currentArrayLength,sizeof(char));
			/*copy over the memory from the currentArray to the destination*/
			memcpy(arrayToSet,currentArray,currentArrayLength*sizeof(char));
			/*now set the length of this array*/
			((args->stringArgs)->arrayLengths)[arrayIndex] = currentArrayLength;
			/*finally point the stringArgs to this*/
			((args->stringArgs)->theArrays)[arrayIndex] = arrayToSet;
		}
	}
	else
	{
		/*the user passed in a null pointer, so set the internal pointer to null*/
		args->stringArgs = 0;
	}
}


/**
    setLongArrayArgArray - sets an array of arrays of longs arguments

    @param longArrayArray - the long2DArray struct containing the long arrays
*/
void setLongArrayArgArray( arguments * args, long2DArray * longArrayArray)
{
	/*before setting anything we need to check the previous longArrayArgs*/
	if(args->longArrayArgs)
	{
		/*then it was previously allocated, so we need to free it*/
		safeDeleteLong(args->longArrayArgs);
	}
	/*now check to see if longArrayArray exists*/
	if(longArrayArray)
	{
		/*save this as a local variable so we don't have to call the function multiple times*/
		byte totalArrayNum = getNumArraysLong(longArrayArray);
		/*now we know it is free, we can allocate a new one*/
		args->longArrayArgs = (long2DArray *) calloc(1,sizeof(long2DArray));
		/*because we know that the matrix was just allocated*/
		/*we can optimize this to not have to call the setNumArraysChar function*/
		(args->longArrayArgs)->numArrays = totalArrayNum;
		(args->longArrayArgs)->arrayLengths = (byte *)calloc(totalArrayNum,sizeof(byte));
		(args->longArrayArgs)->theArrays = (long **)calloc(totalArrayNum,sizeof(long *));
		/*also update the number of args structure with this number*/
		setStringArgNum(args,totalArrayNum);
		long * arrayToSet;
		byte arrayIndex;
		long * currentArray;
		byte currentArrayLength;
		/*for each string in the passed struct, add that to the internal 2d array struct*/
		for (arrayIndex = 0; arrayIndex < totalArrayNum; arrayIndex++)
		{
			/*first get the array*/
			currentArray = (longArrayArray->theArrays)[arrayIndex];
			/*now its length*/
			currentArrayLength = (longArrayArray->arrayLengths)[arrayIndex];
			/*now make a new array for the destination*/
			arrayToSet = (long *)calloc(currentArrayLength,sizeof(long));
			/*copy over the memory from the currentArray to the destination*/
			memcpy(arrayToSet,currentArray,currentArrayLength*sizeof(long));
			/*now set the length of this array*/
			((args->longArrayArgs)->arrayLengths)[arrayIndex] = currentArrayLength;
			/*finally point the longArrayArgs to this*/
			((args->longArrayArgs)->theArrays)[arrayIndex] = arrayToSet;
		}
	}
	else
	{
		/*the user passed in a null pointer, so set the internal pointer to null*/
		args->longArrayArgs = 0;
	}
}


/**
    getFloatArrayArgArray - gets an array of arrays of floats arguments

    @param - floatArrayArray - the float2DArray struct containing the float arrays
*/
void setFloatArrayArgArray( arguments * args, float2DArray * floatArrayArray)
{
	/*check to make sure that there wasn't previously an allocated float2DArray*/
	/*before setting anything we need to check the previous floatArrayArgs*/
	if(args->floatArrayArgs)
	{
		/*then it was previously allocated, so we need to free it*/
		safeDeleteFloat(args->floatArrayArgs);
	}
	/*now check to see if floatArrayArray exists*/
	if(floatArrayArray)
	{
		/*save this as a local variable so we don't have to call the function multiple times*/
		byte totalArrayNum = getNumArraysFloat(floatArrayArray);
		/*now we know it is free, we can allocate a new one*/
		args->floatArrayArgs = (float2DArray *) calloc(1,sizeof(float2DArray));
		/*because we know that the matrix was just allocated*/
		/*we can optimize this to not have to call the setNumArraysChar function*/
		(args->floatArrayArgs)->numArrays = totalArrayNum;
		(args->floatArrayArgs)->arrayLengths = (byte *)calloc(totalArrayNum,sizeof(byte));
		(args->floatArrayArgs)->theArrays = (float **)calloc(totalArrayNum,sizeof(float *));
		/*also update the number of args structure with this number*/
		setStringArgNum(args,totalArrayNum);
		float * arrayToSet;
		byte arrayIndex;
		float * currentArray;
		byte currentArrayLength;
		/*for each string in the passed struct, add that to the internal 2d array struct*/
		for (arrayIndex = 0; arrayIndex < totalArrayNum; arrayIndex++)
		{
			/*first get the array*/
			currentArray = (floatArrayArray->theArrays)[arrayIndex];
			/*now its length*/
			currentArrayLength = (floatArrayArray->arrayLengths)[arrayIndex];
			/*now make a new array for the destination*/
			arrayToSet = (float *)calloc(currentArrayLength,sizeof(float));
			/*copy over the memory from the currentArray to the destination*/
			memcpy(arrayToSet,currentArray,currentArrayLength*sizeof(float));
			/*now set the length of this array*/
			((args->floatArrayArgs)->arrayLengths)[arrayIndex] = currentArrayLength;
			/*finally point the floatArrayArgs to this*/
			((args->floatArrayArgs)->theArrays)[arrayIndex] = arrayToSet;
		}
	}
	else
	{
		/*the user passed in a null pointer, so set the internal pointer to null*/
		args->floatArrayArgs = 0;
	}
}


/** safeDeleteArguments - safely frees all the memory allocated to a given 
 *      arguments struct
 * 
 * @param args the arguments struct to safely free from memory
 */
void safeDeleteArguments(arguments * args)
{
	/*make sure that args exists before trying to free anything*/
	if(args)
	{
		/*we need to free the longArgs pointer, and the floatArgs pointer, then*/
		/*we need to also free the 2d array structs, but we can just call the safely*/
		/*delete function for those*/
		free(args->floatArgs);
		free(args->longArgs);
		safeDeleteChar(args->stringArgs);
		safeDeleteFloat(args->floatArrayArgs);
		safeDeleteLong(args->longArrayArgs);
		/*lastly, we free the struct memory itself, and null the pointer if it was dynamically allocated*/
		if(isPointerDynamic(args))
		{
			free(args);
		}
	}
}
    
    



#endif	/* ARGUMENTS_H */

/**
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
*/



/* 
 * File:   task.h
 * Author: Ian
 *
 * Created on January 12, 2015, 7:49 PM
 */

#ifndef TASK_H
#define	TASK_H



/*this is for reading in th raw bytes that corresponds to a IEEE 32-bit single precision*/
/*float, and saving it as an unsigned long, then reinterpreting those bytes as an actual float*/


/**
    setID - sets the ID

    An ID of 0 represents a config, read, or write task. Any other ID 
    represents which user supplied function should be run for this
    task

    @param idNum - the identification number of the task
*/
inline void setID( task * theTask, byte idNum)
{
    theTask->id = idNum;
}


/** setPortNumber - sets the port that is to be read from, written to, or 
 *      configured
 * 
 * @param theTask - the task to configure
 * @param portNum - the port number of this task
 */
inline void setPortNumber(task * theTask, byte portNum)
{
    theTask->portNumber = portNum;
}


/** setPinNumber - sets the pin that is to be read from, written to, or 
 *      configured
 * 
 * @param theTask - the task to configure 
 * @param pinNum - the pin number of this task
 */
inline void setPinNumber(task * theTask, byte pinNum)
{
    theTask->pinNumber = pinNum;
}


/** setPinValue - sets the value of the pin that is to be written to
 * 
 * @param theTask - the task to configure
 * @param pinVal - the value of the pin to write
 */
inline void setValue(task * theTask, int pinVal)
{
    theTask->value = pinVal;
}


/** getPinValue - gets the value to be written to
 * 
 * @param theTask - the task to configure
 * @return - the value of the pin that is being written
 */
inline int getValue(task * theTask)
{
    return theTask->value;
}


/** getPinNumber - gets the pin number
 * 
 * @param theTask - the task to configure
 * @return  - the pin number
 */
inline byte getPinNumber(task * theTask)
{
    return theTask->pinNumber;
}


/** getPortNumber - gets the ports
 * 
 * @param theTask - the task to configure
 * @return - the port number
 */
inline byte getPortNumber(task * theTask)
{
    return theTask->portNumber;
}


/** hasArgs - returns 1 if the task has arguments, 0 if it doesn't
 * 
 * @param theTask the task to check
 * @return boolean if the task has arguments or not
 */
inline byte hasArgs(task * theTask)
{
    return getArgNum(theTask->taskArgs)>0;
}


/**
    setType - sets the type of the task

    There are four types of tasks, config task, write task, read task,
    and a function call task. This may be expanded in a future release,
    to include functionality such as I2C and SPI reads and writes and 
    such.

    Type 0 -> Config task, this sets the state of pins and such
    Type 1 -> Digital Write task, this will write a given value to a given pin
    Type 2 -> Digital Read task, this will read the value of the digital pin
 *  Type 3 -> Analog Read task, this will read an analog value from a pin
 *  Type 4 -> Analog Write task, this will write an analog value to pin
    Type 5 -> Function call task, this will call a user supplied function that 
 *      is already present on the Arduino.
 *  Type 6 -> Function delete task, this will delete a running or a scheduled
 *      function call task

    @param typeNum - the type of the task
*/
inline void setType( task * theTask, byte typeNum)
{
    theTask->type = typeNum;
}


/**
    setArgs - sets the arguments of the task if it is a function task

    @param theArgs - the arguments struct that contains the argument information
*/
void setArgs( task * theTask, arguments * theArgs)
{
	/*make sure the task exists first*/
	if(theTask)
	{
		if(theTask->taskArgs)
		{
			/*then this task previously had args, so we have to free them first*/
			safeDeleteArguments(theTask->taskArgs);
		}
		/*only copy anything over if we have to, if theArgs is null, then don't copy anything and just set theTask->taskArgs to null*/
		if(theArgs)
		{
			/*now we know that taskArgs has been safely freed and we have arguments to deal with, we can allocate a new args structure*/
			theTask->taskArgs = (arguments *) calloc(1,sizeof(arguments));
			/*check to make sure that the allocation worked*/
			if(theTask->taskArgs)
			{
				/*save these as local variables to prevent calling the function multiple times*/
				byte longArgNum = getLongArgNum(theArgs);
				byte floatArgNum = getFloatArgNum(theArgs);
				byte stringArgNum = getStringArgNum(theArgs);
				byte longArrayArgNum = getLongArrayArgNum(theArgs);
				byte floatArrayArgNum = getFloatArrayArgNum(theArgs);
				/*for the number of arguments, we can just call the setter with the getter */
				/*as an argument*/
				setLongArgNum(theTask->taskArgs,longArgNum);
				setFloatArgNum(theTask->taskArgs,floatArgNum);
				setStringArgNum(theTask->taskArgs,stringArgNum);
				setLongArrayArgNum(theTask->taskArgs,longArrayArgNum);
				setFloatArrayArgNum(theTask->taskArgs,floatArrayArgNum);
				/*for each of the argument types, just use the pointer from inside the structure as the source*/
				if(longArgNum)
				{
					setLongArgArray(theTask->taskArgs,theArgs->longArgs,longArgNum);
				}
				if(floatArgNum)
				{
					setFloatArgArray(theTask->taskArgs,theArgs->floatArgs,floatArgNum);
				}
				if(stringArgNum)
				{
					setStringArgArray(theTask->taskArgs,theArgs->stringArgs);
				}
				if(longArrayArgNum)
				{
					setLongArrayArgArray(theTask->taskArgs,theArgs->longArrayArgs);
				}
				if(floatArrayArgNum)
				{
					setFloatArrayArgArray(theTask->taskArgs,theArgs->floatArrayArgs);
				}
			}
		}
		else
		{
			/*the user passed in a null pointer, so set the arguments as that*/
			theTask->taskArgs = 0;
		}
	}
}


/**
    setInitialWaitTime - sets the amount of time in milliseconds to delay
            execution of the task if the task is a function call

    @param initialWaitTime - the time to delay execution
*/
inline void setInitialWaitTime( task * theTask, unsigned long initialWaitTime)
{
    theTask->waitTime = initialWaitTime;
}


/**
    setIterationCount - sets the number of times that a task should run

    @param iterationCount - the number of times the task should be run
*/
inline void setIterationCount( task * theTask, unsigned long iterationCount)
{
    theTask->numTimesToRun = iterationCount;
}


/**
    setSyncTime - sets the amount of time to wait in milliseconds
            between function calls if there are multiple iterations

    @param syncTimeNum - the amount of time to wait between iterations
*/
inline void setSyncTime( task * theTask, unsigned long syncTimeNum)
{
    theTask->syncTime = syncTimeNum;
}


/**
    setRunTimeLength - sets the amount of total time that should be spent
            running a given task

    @param runTime - the amount of time that a task should run for
*/
inline void setRunTimeLength( task * theTask, unsigned long runTime)
{
    theTask->timeToRun = runTime;
}


/** setLastTimeUpdated will set the last time that the task was updated with
 *      respect to its timer to run the function with the new time
 * 
 * @param theTask - the task to configure
 * @param newTime - the newTime to set update it to
 */
inline void setLastTimeUpdated( task * theTask, unsigned long newTime)
{
    theTask->lastTimeUpdated = newTime;
}


/**
    setArgSignature - sets the order of the arguments passed to the function
            call

    @param signature - a string corresponding to the order and type of the
            arguments passed to the function
    @param numArgs - the number of arguments for the function
*/
void setArgSignature( task * theTask, char * signature, byte numArgs)
{
    /*first check to see if the internal pointer is a null pointer, if it*/
    /*isn't, then we have to free the memory that is currently referenced by it*/
    if (theTask->argSignature)
    {
        free(theTask->argSignature);
    }
	/*first check to see if we even need to allocate for the arg signature*/
	if(numArgs)
	{
		/*malloc some memory for the internal signature*/
		theTask->argSignature = (char *) calloc(numArgs,sizeof(char));
		/*check to make sure that malloc didn't fail by checking for a*/
		/*null pointer*/
		if (theTask->argSignature)
		{
			/*allocated successfully, we can now copy the user's string*/
			memcpy(theTask->argSignature,signature,sizeof(char)*numArgs);
		}
	}
	else
	{
		/*null the pointer as the user says that there aren't any arguments*/
		theTask->argSignature =0;
	}
}

/** getID - gets the ID
 * 
 */
inline byte getID( task * theTask)
{
    return theTask->id;
}


/** getType - sets the type of the task
 * 
 */
inline byte getType( task * theTask)
{
    return theTask->type;
}


/** getArgs - gets the arguments struct of the task
 * 
 */
arguments * getArgs( task * theTask)
{	
    /*we will just create a new arguments structure and pass that back to the user*/
	/*allocate memory for the new arguments*/
    arguments * returnArgs = (arguments *) calloc(1,sizeof(arguments));
	if(returnArgs)
	{
		/*save these as local variables to prevent calling the function multiple times*/
		byte longArgNum = getLongArgNum(theTask->taskArgs);
		byte floatArgNum = getFloatArgNum(theTask->taskArgs);
		byte stringArgNum = getStringArgNum(theTask->taskArgs);
		byte longArrayArgNum = getLongArrayArgNum(theTask->taskArgs);
		byte floatArrayArgNum = getFloatArrayArgNum(theTask->taskArgs);
		/*for the number of arguments, we can just call the setter with the getter */
		/*as an argument*/
		setLongArgNum(returnArgs,longArgNum);
		setFloatArgNum(returnArgs,floatArgNum);
		setStringArgNum(returnArgs,stringArgNum);
		setLongArrayArgNum(returnArgs,longArrayArgNum);
		setFloatArrayArgNum(returnArgs,floatArrayArgNum);
		/*for each of the argument types, use the structure member pointer as the source*/
		if(longArgNum)
		{
			setLongArgArray(returnArgs,(theTask->taskArgs)->longArgs,longArgNum);
		}
		if(floatArgNum)
		{
			setFloatArgArray(returnArgs,(theTask->taskArgs)->floatArgs,floatArgNum);
		}
		if(stringArgNum)
		{
			setStringArgArray(returnArgs,(theTask->taskArgs)->stringArgs);
		}
		if(longArrayArgNum)
		{
			setLongArrayArgArray(returnArgs,(theTask->taskArgs)->longArrayArgs);
		}
		if(floatArrayArgNum)
		{
			setFloatArrayArgArray(returnArgs,(theTask->taskArgs)->floatArrayArgs);
		}
		/*finally return the new arguments*/
		return returnArgs;
	}
	else
	{
		/*memory allocation failed, so return null*/
		return 0;
	}
}


/** getInitialWaitTime - the amount of time to wait before
 * 
 * @param theTask
 * @return the time before the task should be run in millis
 */
inline unsigned long getInitialWaitTime( task * theTask)
{
    return theTask->waitTime;
}


/** getIterationCount - gets the number of times that the task should be run for
 * 
 * @param theTask the task to work on
 * @return the number of times to run, or the iteration count
 */
inline unsigned long getIterationCount(task * theTask)
{
    return theTask->numTimesToRun;
}


/** getSyncTime - gets the amount of time in milliseconds that should elapse in
 *      between function calls
 * 
 * @param theTask - the task to work on
 * @return the amount of time in milliseconds between function calls
 */
inline unsigned long getSyncTime( task * theTask)
{
    return theTask->syncTime;
}


/** getRunTimeLength - gets the total amount of time in milliseconds that should
 *      be spent running a task.
 * 
 * @param theTask - the task to work on
 * @return the amount of time in milliseconds after which the function is no 
 *      longer called
 */
inline unsigned long getRunTimeLength( task * theTask)
{
    return theTask->timeToRun;
}


/** getLastTimeUpdated will return the time in milliseconds after the arduino
 *      started keeping track of time that the time before one should run the 
 *      function
 * 
 * @param theTask
 * @return 
 */
inline unsigned long getLastTimeUpdated( task * theTask)
{
    return theTask->lastTimeUpdated;
}



/** getArgSignature - gets the argument signature from the task
 * 
 * @param theTask - the task to work on
 * @return a string corresponding to the arguments' order and type for the 
 *      function call
 */
char * getArgSignature( task * theTask)
{
    /*make a new pointer so that the internal string isn't modified by the user*/
    /*inadvertently*/
    char * returnPointer = 0;
	if(getArgNum(theTask->taskArgs))
	{
		byte argNum = getArgNum(theTask->taskArgs);
		returnPointer = (char *) calloc(argNum,sizeof(char));
		/*check if the memory was properly allocated*/
		if (returnPointer)
		{
			/*we have the memory, so copy over the internal array to this new pointer*/
			memcpy(returnPointer,theTask->argSignature,(argNum+1)*sizeof(char));
		}
	}
	return returnPointer;
}



/** taskCopy will copy the contents of the sourceTask pointer to the destination
 *      task pointer
 * 
 *  Note: the destinationTask pointer must not be null, new memory will not be
 *      allocated
 * 
 *  Note: the sourceTask pointer is NOT safely deleted and deallocated in this
 *      function, that MUST be done elsewhere
 * 
 * @param destinationTask - the task pointer to copy to
 * @param sourceTask - the task pointer to copy from
 */
void taskCopy(task * destinationTask, task * sourceTask)
{
    arguments * taskArgs = 0;
	/*get the arguments from the task*/
    taskArgs = getArgs(sourceTask);
	/*set the arguments*/
    setArgs(destinationTask,taskArgs);
	/*now safely delete the taskArgs*/
	safeDeleteArguments(taskArgs);
    
	/*char * argSignature;*/
	/*now get the arg signature*/
    /*argSignature = getArgSignature(sourceTask);*/
	/*set the arg signature*/
    /*setArgSignature(destinationTask,argSignature,strlen(argSignature)+1);*/
    /*we can now free the argSignature*/
    /*free(argSignature);*/
    /*for all the rest of the fields, we can just set the field with the result*/
    /*of the getter from the source*/
    setLastTimeUpdated(destinationTask,getLastTimeUpdated(sourceTask));
    setSyncTime(destinationTask,getSyncTime(sourceTask));
    setRunTimeLength(destinationTask,getRunTimeLength(sourceTask));
    setSyncTime(destinationTask,getSyncTime(sourceTask));
    setIterationCount(destinationTask,getIterationCount(sourceTask));
    setInitialWaitTime(destinationTask,getInitialWaitTime(sourceTask));
    setType(destinationTask,getType(sourceTask));
    setID(destinationTask,getID(sourceTask));
    setPortNumber(destinationTask,getPortNumber(sourceTask));
    setPinNumber(destinationTask,getPinNumber(sourceTask));
    setValue(destinationTask,getValue(sourceTask));
}




/** taskUpdate - update the initalWaitTime of a task
 * 
 *  Note: uses millis() from the Arduino library to determine the global clock
 * 
 * @param theTask - the task to configure
 */
void taskUpdate(task * theTask)
{
	/*save these as local variables to prevent calling function multiple times*/
	unsigned long lastTimeUpdated = getLastTimeUpdated(theTask);
	unsigned long initialWaitTime = getInitialWaitTime(theTask);
    /*check if the task time is more than 0*/
    if (initialWaitTime)
    {
        /*the wait time is at least 1, check if subtracting the elapsed time */
        /*will make the time go negative, as the time is stored as an unsigned*/
        /*long, so we can't try to actually subtract it and make it overflow to*/
        /*a huge number, so get the elapsed time since the task was last updated*/
        /*and compare it rather than subtract it here*/
		/*first check if getLastTimeUpdated is zero, if it is, then this task has never*/
		/*been updated before, and we have to set the last time it was updated to the current*/
		/*time and just return*/
		if(lastTimeUpdated==0)
		{
			setLastTimeUpdated(theTask,millis());
			return;
		}
        unsigned long elapsedTime = millis() - lastTimeUpdated;
        if (elapsedTime >= initialWaitTime)
        {
            /*overflow would occur, so just set the wait time to be 0*/
            setInitialWaitTime(theTask,0);
            return;
        }
        else
        {
            /*decrement the waitTime by the elapsed time*/
            setInitialWaitTime(theTask,(initialWaitTime - elapsedTime));
			/*update the last time updated to the current time*/
			setLastTimeUpdated(theTask,millis());
            return;
        }
    }
    else
    {
        /*the task's wait time is 0, hence it is ready to run and we don't need*/
        /*to update anything and we can just return*/
        return;
    }
}


/** safeDeleteTask - safely deletes a task by deallocating all the relevant 
 *      memory referenced by the task
 * 
 * @param theTask - the task to safely delete
 */
void safeDeleteTask(task * theTask)
{
	/*first check to make sure theTask exists, if not then don't do anything*/
	if(theTask)
	{
		/*to safely delete the task, we need to free the following pointer:*/
		/*argSignature - just points to one string*/
		/*we also need to free the memory asociated with the taskArgs, but we can*/
		/*just call the safeDeleteArguments function on that*/
		free(theTask->argSignature);
		safeDeleteArguments(theTask->taskArgs);
		/*we can finally free and null the pointer if it was dynamicaly generated*/
		if(isPointerDynamic(theTask))
		{
			free(theTask);
		}
	}
}



/*sends digital read port value over the serial line*/
void firmataDigitalReadPacketSend(byte port, byte value)
{
	/*first write the digital port header, with the port as the low nibble*/
	SerialLink.write(DIGITAL_WRITE_TASK | (port & 15));
	/*then write the least significant 7 bits of the value*/
	SerialLink.write(value & 127);
	/*finally write the most significant bit of the value*/
	SerialLink.write((value & 128)>>7);
	/*finally, flush the serial buffer*/
	SerialLink.flush();
}

/*sends analog read pin value over the serial line*/
void firmataAnalogReadPacketSend(byte pin, int value)
{
	/*first write the analog pin header, with the pin as the low nibble*/
	SerialLink.write(ANALOG_WRITE_TASK | pin);
	/*then the least significant 7 bits of the value*/
	SerialLink.write(value & 127);
	/*finally write the most significant 7 bits*/
	SerialLink.write((value & 16256)>>7);
	/*finally, flush the serial buffer*/
	SerialLink.flush();
}


/*sends a floating point number over the serial line*/
void firmataTaskFloatSend(float data)
{
	/*first write the sysex start header*/
	SerialLink.write(SYSEX_START);
	/*then write the float number header*/
	SerialLink.write(FLOAT_NUM);
	/*then convert the float into raw bytes*/
	Converter convert;
	convert.theFloat = data;
	byte first = (convert.rawBytes )>> 24;
	byte second = (convert.rawBytes & 16711680) >> 16;
	byte third = (convert.rawBytes & 65280) >> 8;
	byte fourth = convert.rawBytes & 255;
	/*now write those bytes in order*/
	SerialLink.write(first);
	SerialLink.write(second);
	SerialLink.write(third);
	SerialLink.write(fourth);
	/*send sysex end byte*/
	SerialLink.write(SYSEX_END);
	/*finally, flush the serial buffer*/
	SerialLink.flush();
}


/*sends a long number over the serial line*/
void firmataTaskLongSend(long data)
{
	/*first write the sysex start header*/
	SerialLink.write(SYSEX_START);
	/*then write the long number header*/
	SerialLink.write(LONG_NUM);
	/*then convert the float into raw bytes*/
	Converter convert;
	convert.theLong = data;
	byte first = (convert.rawBytes )>> 24;
	byte second = (convert.rawBytes & 16711680) >> 16;
	/*16711680 is 255 bitshifted up 16*/
	byte third = (convert.rawBytes & 65280) >> 8;
	/*65280 is 255 bitshifted up 8*/
	byte fourth = convert.rawBytes & 255;
	/*now write those bytes in order*/
	SerialLink.write(first);
	SerialLink.write(second);
	SerialLink.write(third);
	SerialLink.write(fourth);
	/*send sysex end byte*/
	SerialLink.write(SYSEX_END);
	/*finally, flush the serial buffer*/
	SerialLink.flush();
} 


/*sends a string over the serial line*/
void firmataTaskStringSend(char * data)
{
	/*first, get the length of the string*/
	int stringLength = strlen(data);
	/*now write the sysex start byte*/
	SerialLink.write(SYSEX_START);
	/*now write the string identifier byte*/
	SerialLink.write(STRING);
	/*now write how many bytes are in the string as 2 bytes*/
	SerialLink.write(stringLength & 127);
	SerialLink.write((stringLength & 16256) >> 7);
	/*now iterate through the string, writing each char on the serial buffer*/
	int charIndex = 0;
	for(;charIndex < stringLength;charIndex++)
	{
		SerialLink.write(data[charIndex]);
	}
	/*write a sysex end byte*/
	SerialLink.write(SYSEX_END);
	/*finally, flush the serial buffer*/
	SerialLink.flush();
}
	



/*copied from the Arduino libraries - specifically the wiring_digital.c library*/
/*(it's a private method in that file, and I need it to avoid having to use*/
/*digitalRead*/
static inline void turnOffPWM(uint8_t timer)
{
	switch (timer)
	{
		#if defined(TCCR1A) && defined(COM1A1)
		case TIMER1A:   
			cbi(TCCR1A, COM1A1);    
			break;
		#endif
		#if defined(TCCR1A) && defined(COM1B1)
		case TIMER1B:   
			cbi(TCCR1A, COM1B1);    
			break;
		#endif
		
		#if defined(TCCR2) && defined(COM21)
		case  TIMER2:   
			cbi(TCCR2, COM21);      
			break;
		#endif
		
		#if defined(TCCR0A) && defined(COM0A1)
		case  TIMER0A:  
			cbi(TCCR0A, COM0A1);    
			break;
		#endif
		
		#if defined(TIMER0B) && defined(COM0B1)
		case  TIMER0B:  
			cbi(TCCR0A, COM0B1);    
			break;
		#endif
		#if defined(TCCR2A) && defined(COM2A1)
		case  TIMER2A:  
			cbi(TCCR2A, COM2A1);    
			break;
		#endif
		#if defined(TCCR2A) && defined(COM2B1)
		case  TIMER2B:  
			cbi(TCCR2A, COM2B1);    
			break;
		#endif
		
		#if defined(TCCR3A) && defined(COM3A1)
		case  TIMER3A:  
			cbi(TCCR3A, COM3A1);    
			break;
		#endif
		#if defined(TCCR3A) && defined(COM3B1)
		case  TIMER3B:  
			cbi(TCCR3A, COM3B1);    
			break;
		#endif
		#if defined(TCCR3A) && defined(COM3C1)
		case  TIMER3C:  
			cbi(TCCR3A, COM3C1);    
			break;
		#endif

		#if defined(TCCR4A) && defined(COM4A1)
		case  TIMER4A:  
			cbi(TCCR4A, COM4A1);    
			break;
		#endif					
		#if defined(TCCR4A) && defined(COM4B1)
		case  TIMER4B:  
			cbi(TCCR4A, COM4B1);    
			break;
		#endif
		#if defined(TCCR4A) && defined(COM4C1)
		case  TIMER4C:  
			cbi(TCCR4A, COM4C1);    
			break;
		#endif			
		#if defined(TCCR4C) && defined(COM4D1)
		case TIMER4D:	
			cbi(TCCR4C, COM4D1);	
			break;
		#endif			
			
		#if defined(TCCR5A)
		case  TIMER5A:  
			cbi(TCCR5A, COM5A1);    
			break;
		case  TIMER5B:  
			cbi(TCCR5A, COM5B1);    
			break;
		case  TIMER5C:  
			cbi(TCCR5A, COM5C1);    
			break;
		#endif
	}
}




/*for checking if the pwm timer on a pin is on or not, */
/*used for digital reading and writing to determine if writing to */
/*or reading from a pwm pin that has the pwm timer on is intended*/
byte PWMOn(byte pin)
{
	switch(digitalPinToTimer(pin))
	{
		#if defined(TCCR0) && defined(COM00) && !defined(__AVR_ATmega8__)
		case TIMER0A:
			// check pwm by checking on timer 0
			return getBitValue(TCCR0, COM00);
		#endif

		#if defined(TCCR0A) && defined(COM0A1)
		case TIMER0A:
			// check pwm by checking on timer 0, channel A
			return getBitValue(TCCR0A,COM0A1);
		#endif

		#if defined(TCCR0A) && defined(COM0B1)
		case TIMER0B:
			// check pwm by checking on timer 0, channel B
			return getBitValue(TCCR0A, COM0B1);
		#endif

		#if defined(TCCR1A) && defined(COM1A1)
		case TIMER1A:
			// check pwm by checking on timer 1, channel A
			return getBitValue(TCCR1A, COM1A1);
		#endif

		#if defined(TCCR1A) && defined(COM1B1)
		case TIMER1B:
			// check pwm by checking on timer 1, channel B
			return getBitValue(TCCR1A, COM1B1);
		#endif

		#if defined(TCCR2) && defined(COM21)
		case TIMER2:
			// check pwm by checking on timer 2
			return getBitValue(TCCR2, COM21);
		#endif

		#if defined(TCCR2A) && defined(COM2A1)
		case TIMER2A:
			// check pwm by checking on timer 2, channel A
			return getBitValue(TCCR2A, COM2A1);
		#endif

		#if defined(TCCR2A) && defined(COM2B1)
		case TIMER2B:
			// check pwm by checking on timer 2, channel B
			return getBitValue(TCCR2A, COM2B1);
		#endif

		#if defined(TCCR3A) && defined(COM3A1)
		case TIMER3A:
			// check pwm by checking on timer 3, channel A
			return getBitValue(TCCR3A, COM3A1);
		#endif

		#if defined(TCCR3A) && defined(COM3B1)
		case TIMER3B:
			// check pwm by checking on timer 3, channel B
			return getBitValue(TCCR3A, COM3B1);
		#endif

		#if defined(TCCR3A) && defined(COM3C1)
		case TIMER3C:
			// check pwm by checking on timer 3, channel C
			return getBitValue(TCCR3A, COM3C1);
		#endif

		#if defined(TCCR4A)
		case TIMER4A:
			// check pwm by checking on timer 4, channel A
			return getBitValue(TCCR4A, COM4A1);
		#endif
		
		#if defined(TCCR4A) && defined(COM4B1)
		case TIMER4B:
			// check pwm by checking on timer 4, channel B
			return getBitValue(TCCR4A, COM4B1);
		#endif

		#if defined(TCCR4A) && defined(COM4C1)
		case TIMER4C:
			// check pwm by checking on timer 4, channel C
			return getBitValue(TCCR4A, COM4C1);
		#endif
			
		#if defined(TCCR4C) && defined(COM4D1)
		case TIMER4D:				
			// check pwm by checking on timer 4, channel D
			return getBitValue(TCCR4C, COM4D1);
		#endif

						
		#if defined(TCCR5A) && defined(COM5A1)
		case TIMER5A:
			// check pwm by checking on timer 5, channel A
			return getBitValue(TCCR5A, COM5A1);
		#endif

		#if defined(TCCR5A) && defined(COM5B1)
		case TIMER5B:
			// check pwm by checking on timer 5, channel B
			return getBitValue(TCCR5A, COM5B1);
		#endif

		#if defined(TCCR5A) && defined(COM5C1)
		case TIMER5C:
			// check pwm by checking on timer 5, channel C
			return getBitValue(TCCR5A, COM5C1);
		#endif
		
		/*case for not pwm pin is 0, there is no timer associated with non-pwm pins*/
		case NOT_ON_TIMER:
		default:
			return 0;
	}
}







void configTaskRun(task * configTask, byte * pinConfigurations)
{
	/*all we need to do is make sure that the config is either input (0) or output (1), and that we 
	don't try and configure the serial pins (0 and 1) or a pin that doesn't exist, and we then also 
	set the pinConfigurations bit in the corresponding byte*/
	byte pin = getPinNumber(configTask);
	byte config = getValue(configTask);
	/*need to set the pin configuration bit for that pin in the port, to represent the fact that*/
	/*the pin has now been hard configured, so the system doesn't flip flop it for the user*/
	if(pin > 1 && pin <20 && (config == 0 || config ==1) )
	{
		pinMode(pin,config);
		pinConfigurations[pin / 8] |= _BV(pin % 8);
	}
	return;
}





void readTaskRun(task * readTask, byte * pinConfigurations)
{
#if defined (ARDUINO_AVR_UNO)
    /*********************************************************************/
    /*********************************************************************/
    
    /***********         ARDUINO UNO       *******************************/
    
    /*********************************************************************/
    /*********************************************************************/
	/*need to check which kind of read we are doing*/
	if(getType(readTask) == ANALOG_READ)
	{
		/*analog read uses a pin*/
		byte pin = getPinNumber(readTask);
		/*for analog pins, we know that the port is PORTC, or firmata port 2, so first we check to see if*/
		/*the pin has been hard configured, and if it has, whether or not it was hard configued for input*/
		/*the boolean logic is as follows:*/
		/* data direction bit | hard configured? | allow read?*/
		/*        0           |        0         |      1*/
		/*        0           |        1         |      1*/
		/*        1           |        0         |      1*/
		/*        1           |        1         |      0*/
		/*this yields a logical nand, or ~(data direction & hard configured)*/
		/*since we alread know the port because the pin is an analog pin, we don't need to switch on the port*/
		/*_BV(X) is equivalent to 1<<X, this is defined in avr/io.h*/
		/*DDRC & _BV(pin % 8) is equivalent to the direction of the analog pin in question, with 1 being output and 0 being input*/
		if(!((pinConfigurations[2] & pin % 8) && (DDRC & _BV(pin % 8))))
		{
			/*we can read from the pin, as it is not been hard configured as an output*/
			/*before performing an analog read, we need to check to see if a 1 was recently */
			/*written to the device, if it was, then we need to wait for 35 milliseconds before*/
			/*sending the value back, this is because the ADC needs to "settle in"*/
			/*see bug 292046*/
			if ((DDRC & _BV(pin % 8)) && (PORTC & _BV(pin % 8)))
			{
				/*35 ms seems to be just the right amount of time for the ADC to settle*/
				delay(35);
			}
			/*we can now read from the pin, so read the value and send it back in a firmata packet*/
			/*the pin sent over the wire is a number between 0 and 5, so to turn that into */
			/*arduino pin numbers, add 14 to the pin number*/
			pinMode(pin+14,INPUT);
			firmataAnalogReadPacketSend(pin,analogRead(pin));
		}
		return;
	}
	else
	{
		/*digital read uses a port */
		byte port = getPortNumber(readTask);
		byte PWMDisable = getPinNumber(readTask);
		register byte value = 0;
		/*switch on port, and for each pin in the port, if we should read from it, add it to the bitmask*/
		/*and return the bitmask to the client using the Firmata protocol*/
		/*also, we are using the boolean expression developed above for analog pin reading, but*/
		/*see writeTaskRun documentation for specifics of other details, it basically is just*/
		/*making sure that we don't read from pins that the user explicitly said were*/
		/*to be output*/
		/*there's also no danger from being interrupted before reading the value, the worst case scenario is that*/
		/*the value reported is ever so slightly out of date, which can be solved with another request*/
		switch (port)
		{
			case 0:
				/*PORTD (note we ignore pins 0 and 1 as those are for SerialLink communication)*/
				if(!((pinConfigurations[0] & 4) && (DDRD & 4) ))
				{
					/*digital pin 2*/
					/*the pin is good to be read from, so we need to set it as input, then read it and add the*/
					/*value into the value bitmask*/
					/*set the mode as input*/
					DDRD &= ~(4);
					/*read the input into the value bitmask*/
					value |= ((PIND & 4) ? 4 : 0);
				}
				if(PWMOn(3))
				{
					/*digital pin 3*/
					/*PWM timer is on this pin, so check the disable bit*/
					if(PWMDisable & 1)
					{
						if(!((pinConfigurations[0] & 8) && (DDRD & 8) ))
						{
							/*digital pin 3 (PWM pin)*/
							/*the pin is good to be read from*/
							/*before we read from this pin, because it is a possible pwm pin, and may have previously been in */
							/*a PWM state, we need to turn off PWM on this pin*/
							turnOffPWM(digitalPinToTimer(3));
							/*before reading, set pin as input, then read it and add the value into the value bitmask*/
							/*set the mode as input*/
							DDRD &= ~(8);
							/*read the input into the value bitmask*/
							value |= ((PIND & 8) ? 8 : 0);
						}
					}
				}
				else if(!((pinConfigurations[0] & 8) && (DDRD & 8) ))
				{
					/*digital pin 5 (PWM pin)*/
					/*the pin is good to be read from*/
					/*note don't need to turn off PWM here*/
					/*before reading, set pin as input, then read it and add the value into the value bitmask*/
					/*set the mode as input*/
					DDRD &= ~(8);
					/*read the input into the value bitmask*/
					value |= ((PIND & 8) ? 8 : 0);
				}
				if(!((pinConfigurations[0] & 16) && (DDRD & 16) ))
				{
					/*digital pin 4*/
					/*the pin is good to be read from, so we need to set it as input, then read it and add the*/
					/*value into the value bitmask*/
					/*set the mode as input*/
					DDRD &= ~(16);
					/*read the input into the value bitmask*/
					value |= ((PIND & 16) ? 16 : 0);
				}
				if(PWMOn(5))
				{
					/*digital pin 5*/
					/*PWM timer is on this pin, so check the disable bit*/
					if(PWMDisable & 2)
					{
						if(!((pinConfigurations[0] & 32) && (DDRD & 32) ))
						{
							/*digital pin 5 (PWM pin)*/
							/*the pin is good to be read from*/
							/*before we read from this pin, because it is a possible pwm pin, and may have previously been in */
							/*a PWM state, we need to turn off PWM on this pin*/
							turnOffPWM(digitalPinToTimer(5));
							/*before reading, set pin as input, then read it and add the value into the value bitmask*/
							/*set the mode as input*/
							DDRD &= ~(32);
							/*read the input into the value bitmask*/
							value |= ((PIND & 32) ? 32 : 0);
						}
					}
				}
				else if(!((pinConfigurations[0] & 32) && (DDRD & 32) ))
				{
					/*digital pin 5 (PWM pin)*/
					/*the pin is good to be read from*/
					/*note don't need to turn off PWM here*/
					/*before reading, set pin as input, then read it and add the value into the value bitmask*/
					/*set the mode as input*/
					DDRD &= ~(32);
					/*read the input into the value bitmask*/
					value |= ((PIND & 32) ? 32 : 0);
				}
				if(PWMOn(6))
				{
					/*digital pin 6*/
					/*PWM timer is on this pin, so check the disable bit*/
					if(PWMDisable & 4)
					{
						if(!((pinConfigurations[0] & 64) && (DDRD & 64) ))
						{
							/*digital pin 6 (PWM pin)*/
							/*the pin is good to be read from*/
							/*before we read from this pin, because it is a possible pwm pin, and may have previously been in */
							/*a PWM state, we need to turn off PWM on this pin*/
							turnOffPWM(digitalPinToTimer(6));
							/*before reading, set pin as input, then read it and add the value into the value bitmask*/
							/*set the mode as input*/
							DDRD &= ~(64);
							/*read the input into the value bitmask*/
							value |= ((PIND & 64) ? 64 : 0);
						}
					}
				}
				else if(!((pinConfigurations[0] & 64) && (DDRD & 64) ))
				{
					/*digital pin 6 (PWM pin)*/
					/*the pin is good to be read from*/
					/*note don't need to turn off PWM here*/
					/*before reading, set pin as input, then read it and add the value into the value bitmask*/
					/*set the mode as input*/
					DDRD &= ~(64);
					/*read the input into the value bitmask*/
					value |= ((PIND & 64) ? 64 : 0);
				}
				if(!((pinConfigurations[0] & 128) && (DDRD & 128) ))
				{
					/*digital pin 7*/
					/*the pin is good to be read from, so we need to set it as input, then read it and add the*/
					/*value into the value bitmask*/
					/*set the mode as input*/
					DDRD &= ~(128);
					/*read the input into the value bitmask*/
					value |= ((PIND & 128) ? 128 : 0);
				}
				firmataDigitalReadPacketSend(port,value);
				return;
			case 1:
				/*PORTB*/
				if(!((pinConfigurations[1] & 1) && (DDRB & 1) ))
				{
					/*digital pin 8*/
					/*the pin is good to be read from, so we need to set it as input, then read it and add the*/
					/*value into the value bitmask*/
					/*set the mode as input*/
					DDRB &= ~(1);
					/*read the input into the value bitmask*/
					value |= (PINB & 1);
				}
				if(PWMOn(9))
				{
					/*digital pin 9*/
					/*PWM timer is on this pin, so check the disable bit*/
					if(PWMDisable & 1)
					{
						if(!((pinConfigurations[1] & 2) && (DDRB & 2) ))
						{
							/*digital pin 9 (PWM pin)*/
							/*the pin is good to be read from*/
							/*before we read from this pin, because it is a possible pwm pin, and may have previously been in */
							/*a PWM state, we need to turn off PWM on this pin*/
							turnOffPWM(digitalPinToTimer(9));
							/*before reading, set pin as input, then read it and add the value into the value bitmask*/
							/*set the mode as input*/
							DDRB &= ~(2);
							/*read the input into the value bitmask*/
							value |= ((PINB & 2) ? 2 : 0);
						}
					}
				}
				else if(!((pinConfigurations[1] & 2) && (DDRB & 2) ))
				{
					/*digital pin 9 (PWM pin)*/
					/*the pin is good to be read from*/
					/*note don't need to turn off PWM here*/
					/*before reading, set pin as input, then read it and add the value into the value bitmask*/
					/*set the mode as input*/
					DDRB &= ~(2);
					/*read the input into the value bitmask*/
					value |= ((PINB & 2) ? 2 : 0);
				}
				if(PWMOn(10))
				{
					/*digital pin 10*/
					/*PWM timer is on this pin, so check the disable bit*/
					if(PWMDisable & 2)
					{
						if(!((pinConfigurations[1] & 4) && (DDRB & 4) ))
						{
							/*digital pin 10 (PWM pin)*/
							/*the pin is good to be read from*/
							/*before we read from this pin, because it is a possible pwm pin, and may have previously been in */
							/*a PWM state, we need to turn off PWM on this pin*/
							turnOffPWM(digitalPinToTimer(10));
							/*before reading, set pin as input, then read it and add the value into the value bitmask*/
							/*set the mode as input*/
							DDRB &= ~(4);
							/*read the input into the value bitmask*/
							value |= ((PINB & 4) ? 2 : 0);
						}
					}
				}
				else if(!((pinConfigurations[1] & 4) && (DDRB & 4) ))
				{
					/*digital pin 10 (PWM pin)*/
					/*the pin is good to be read from*/
					/*note don't need to turn off PWM here*/
					/*before reading, set pin as input, then read it and add the value into the value bitmask*/
					/*set the mode as input*/
					DDRB &= ~(4);
					/*read the input into the value bitmask*/
					value |= ((PINB & 4) ? 4 : 0);
				}
				if(PWMOn(11))
				{
					/*digital pin 11*/
					/*PWM timer is on this pin, so check the disable bit*/
					if(PWMDisable & 4)
					{
						if(!((pinConfigurations[1] & 8) && (DDRB & 8) ))
						{
							/*digital pin 11 (PWM pin)*/
							/*the pin is good to be read from*/
							/*before we read from this pin, because it is a possible pwm pin, and may have previously been in */
							/*a PWM state, we need to turn off PWM on this pin*/
							turnOffPWM(digitalPinToTimer(11));
							/*before reading, set pin as input, then read it and add the value into the value bitmask*/
							/*set the mode as input*/
							DDRB &= ~(8);
							/*read the input into the value bitmask*/
							value |= ((PINB & 8) ? 8 : 0);
						}
					}
				}
				else if(!((pinConfigurations[1] & 8) && (DDRB & 8) ))
				{
					/*digital pin 11 (PWM pin)*/
					/*the pin is good to be read from*/
					/*note don't need to turn off PWM here*/
					/*before reading, set pin as input, then read it and add the value into the value bitmask*/
					/*set the mode as input*/
					DDRB &= ~(8);
					/*read the input into the value bitmask*/
					value |= ((PINB & 8) ? 8 : 0);					
				}
				if(!((pinConfigurations[1] & 16) && (DDRB & 16) ))
				{
					/*digital pin 12*/
					/*the pin is good to be read from, so we need to set it as input, then read it and add the*/
					/*value into the value bitmask*/
					/*set the mode as input*/
					DDRB &= ~(16);
					/*read the input into the value bitmask*/
					value |= ((PINB & 16) ? 16 : 0);
				}
				if(!((pinConfigurations[1] & 32) && (DDRB & 32) ))
				{
					/*digital pin 13*/
					/*the pin is good to be read from, so we need to set it as input, then read it and add the*/
					/*value into the value bitmask*/
					/*set the mode as input*/
					DDRB &= ~(32);
					/*read the input into the value bitmask*/
					value |= ((PINB & 32) ? 32 : 0);
				}
				/*there is no PC6 (actually, it's the reset pin), and no PC7*/
				firmataDigitalReadPacketSend(port,value);
				return;
			case 2:
				/*PORTC*/
				if(!((pinConfigurations[2] & 1) && (DDRC & 1) ))
				{
					/*analog pin 0*/
					/*the pin is good to be read from, so we need to set it as input, then read it and add the*/
					/*value into the value bitmask*/
					/*set the mode as input*/
					DDRC &= ~(1);
					/*read the input into the value bitmask*/
					value |= (PINC & 1);
				}
				if(!((pinConfigurations[2] & 2) && (DDRC & 2) ))
				{
					/*analog pin 1*/
					/*the pin is good to be read from, so we need to set it as input, then read it and add the*/
					/*value into the value bitmask*/
					/*set the mode as input*/
					DDRC &= ~(2);
					/*read the input into the value bitmask*/
					value |= ((PINC & 2) ? 2 : 0);
				}
				if(!((pinConfigurations[2] & 4) && (DDRC & 4) ))
				{
					/*analog pin 2*/
					/*the pin is good to be read from, so we need to set it as input, then read it and add the*/
					/*value into the value bitmask*/
					/*set the mode as input*/
					DDRC &= ~(4);
					/*read the input into the value bitmask*/
					value |= ((PINC & 4) ? 4 : 0);
				}
				if(!((pinConfigurations[2] & 8) && (DDRC & 8) ))
				{
					/*analog pin 3*/
					/*the pin is good to be read from, so we need to set it as input, then read it and add the*/
					/*value into the value bitmask*/
					/*set the mode as input*/
					DDRC &= ~(8);
					/*read the input into the value bitmask*/
					value |= ((PINC & 8) ? 8 : 0);
				}
				if(!((pinConfigurations[2] & 16) && (DDRC & 16) ))
				{
					/*analog pin 4*/
					/*the pin is good to be read from, so we need to set it as input, then read it and add the*/
					/*value into the value bitmask*/
					/*set the mode as input*/
					DDRC &= ~(16);
					/*read the input into the value bitmask*/
					value |= ((PINC & 16) ? 16 : 0);
				}
				if(!((pinConfigurations[2] & 32) && (DDRC & 32) ))
				{
					/*analog pin 5*/
					/*the pin is good to be read from, so we need to set it as input, then read it and add the*/
					/*value into the value bitmask*/
					/*set the mode as input*/
					DDRC &= ~(32);
					/*read the input into the value bitmask*/
					value |= ((PINC & 32) ? 32 : 0);
				}
				/*there is no PC6 and PC7 (actually those pins are for the clock crystal)*/
				firmataDigitalReadPacketSend(port,value);
				return;
			default:
				/*wrong port, don't do anything, just return*/
				return;
		}
	}
#elif defined (ARDUINO_AVR_YUN)
    /*********************************************************************/
    /*********************************************************************/
    
    /***********         ARDUINO YUN       *******************************/
    
    /*********************************************************************/
    /*********************************************************************/
    /*need to check which kind of read we are doing*/
    if(getType(readTask) == ANALOG_READ)
    {
        /*analog read uses a pin*/
        byte pin = getPinNumber(readTask);
        /*for analog pins, we know that the port is PORTC, or firmata port 2, so first we check to see if*/
        /*the pin has been hard configured, and if it has, whether or not it was hard configued for input*/
        /*the boolean logic is as follows:*/
        /* data direction bit | hard configured? | allow read?*/
        /*        0           |        0         |      1*/
        /*        0           |        1         |      1*/
        /*        1           |        0         |      1*/
        /*        1           |        1         |      0*/
        /*this yields a logical nand, or ~(data direction & hard configured)*/
        /*switch on which pin, unlike other arduino models, the yun's analog pins don't have a linear mapping between the arduino and the actual registers*/
        switch(pin)
        {
            case 0:
                if(!((pinConfigurations[2] & 1) && (DDRF & 128)))
                {
                    /*A0 - pin PF7 */
                    /*we can read from the pin, as it is not been hard configured as an output*/
                    /*before performing an analog read, we need to check to see if a 1 was recently */
                    /*written to the device, if it was, then we need to wait for 35 milliseconds before*/
                    /*sending the value back, this is because the ADC needs to "settle in"*/
                    /*see bug 292046*/
                    if ((DDRF & 128) && (PORTF & 128))
                    {
                        /*35 ms seems to be just the right amount of time for the ADC to settle*/
                        delay(35);
                    }
                    /*we can now read from the pin, so read the value and send it back in a firmata packet*/
                    /*the pin sent over the wire is a number between 0 and 5, so to turn that into */
                    /*arduino pin numbers, add 14 to the pin number*/
                    pinMode(A0,INPUT);
                    firmataAnalogReadPacketSend(A0-14,analogRead(A0));
                }
            case 1:
                if(!((pinConfigurations[2] & 2) && (DDRF & 64)))
                {
                    /*A1 - pin PF6 */
                    /*we can read from the pin, as it is not been hard configured as an output*/
                    /*before performing an analog read, we need to check to see if a 1 was recently */
                    /*written to the device, if it was, then we need to wait for 35 milliseconds before*/
                    /*sending the value back, this is because the ADC needs to "settle in"*/
                    /*see bug 292046*/
                    if ((DDRF & 64) && (PORTF & 64))
                    {
                        /*35 ms seems to be just the right amount of time for the ADC to settle*/
                        delay(35);
                    }
                    /*we can now read from the pin, so read the value and send it back in a firmata packet*/
                    /*the pin sent over the wire is a number between 0 and 5, so to turn that into */
                    /*arduino pin numbers, add 14 to the pin number*/
                    pinMode(A1,INPUT);
                    firmataAnalogReadPacketSend(A1-14,analogRead(A1));
                }
            case 2:
                if(!((pinConfigurations[2] & 4) && (DDRF & 32)))
                {
                    /*A2 - pin PF5 */
                    /*we can read from the pin, as it is not been hard configured as an output*/
                    /*before performing an analog read, we need to check to see if a 1 was recently */
                    /*written to the device, if it was, then we need to wait for 35 milliseconds before*/
                    /*sending the value back, this is because the ADC needs to "settle in"*/
                    /*see bug 292046*/
                    if ((DDRF & 32) && (PORTF & 32))
                    {
                        /*35 ms seems to be just the right amount of time for the ADC to settle*/
                        delay(35);
                    }
                    /*we can now read from the pin, so read the value and send it back in a firmata packet*/
                    /*the pin sent over the wire is a number between 0 and 5, so to turn that into */
                    /*arduino pin numbers, add 14 to the pin number*/
                    pinMode(A2,INPUT);
                    firmataAnalogReadPacketSend(A2-14,analogRead(A2));
                }
            case 3:
                if(!((pinConfigurations[2] & 8) && (DDRF & 16)))
                {
                    /*A3 - pin PF4 */
                    /*we can read from the pin, as it is not been hard configured as an output*/
                    /*before performing an analog read, we need to check to see if a 1 was recently */
                    /*written to the device, if it was, then we need to wait for 35 milliseconds before*/
                    /*sending the value back, this is because the ADC needs to "settle in"*/
                    /*see bug 292046*/
                    if ((DDRF & 16) && (PORTF & 16))
                    {
                        /*35 ms seems to be just the right amount of time for the ADC to settle*/
                        delay(35);
                    }
                    /*we can now read from the pin, so read the value and send it back in a firmata packet*/
                    /*the pin sent over the wire is a number between 0 and 5, so to turn that into */
                    /*arduino pin numbers, add 14 to the pin number*/
                    pinMode(A3,INPUT);
                    firmataAnalogReadPacketSend(A3-14,analogRead(A3));
                }
            case 4:
                if(!((pinConfigurations[2] & 16) && (DDRF & 2)))
                {
                    /*A4 - pin PF1 */
                    /*we can read from the pin, as it is not been hard configured as an output*/
                    /*before performing an analog read, we need to check to see if a 1 was recently */
                    /*written to the device, if it was, then we need to wait for 35 milliseconds before*/
                    /*sending the value back, this is because the ADC needs to "settle in"*/
                    /*see bug 292046*/
                    if ((DDRF & 2) && (PORTF & 2))
                    {
                        /*35 ms seems to be just the right amount of time for the ADC to settle*/
                        delay(35);
                    }
                    /*we can now read from the pin, so read the value and send it back in a firmata packet*/
                    /*the pin sent over the wire is a number between 0 and 5, so to turn that into */
                    /*arduino pin numbers, add 14 to the pin number*/
                    pinMode(A4,INPUT);
                    firmataAnalogReadPacketSend(A4-14,analogRead(A4));
                }
            case 5:
                if(!((pinConfigurations[2] & 32) && (DDRF & 1)))
                {
                    /*A5 - pin PF0 */
                    /*we can read from the pin, as it is not been hard configured as an output*/
                    /*before performing an analog read, we need to check to see if a 1 was recently */
                    /*written to the device, if it was, then we need to wait for 35 milliseconds before*/
                    /*sending the value back, this is because the ADC needs to "settle in"*/
                    /*see bug 292046*/
                    if ((DDRF & 1) && (PORTF & 1))
                    {
                        /*35 ms seems to be just the right amount of time for the ADC to settle*/
                        delay(35);
                    }
                    /*we can now read from the pin, so read the value and send it back in a firmata packet*/
                    /*the pin sent over the wire is a number between 0 and 5, so to turn that into */
                    /*arduino pin numbers, add 14 to the pin number*/
                    pinMode(A5,INPUT);
                    firmataAnalogReadPacketSend(A5-14,analogRead(A5));
                }
        }
    }
    else
    {
        /*digital read uses a port */
        byte port = getPortNumber(readTask);
        byte PWMDisable = getPinNumber(readTask);
        register byte value = 0;
        /*switch on port, and for each pin in the port, if we should read from it, add it to the bitmask*/
        /*and return the bitmask to the client using the Firmata protocol*/
        /*also, we are using the boolean expression developed above for analog pin reading, but*/
        /*see writeTaskRun documentation for specifics of other details, it basically is just*/
        /*making sure that we don't read from pins that the user explicitly said were*/
        /*to be output*/
        /*there's also no danger from being interrupted before reading the value, the worst case scenario is that*/
        /*the value reported is ever so slightly out of date, which can be solved with another request*/
        switch (port)
        {
            case 0:
                if(!((pinConfigurations[0] & 4) && (DDRD & 2) ))
                {
                    /*digital pin 2 - PD1*/
                    /*the pin is good to be read from, so we need to set it as input, then read it and add the*/
                    /*value into the value bitmask*/
                    /*set the mode as input*/
                    DDRD &= ~(4);
                    /*read the input into the value bitmask*/
                    //note that on the yun, the port and firmata pin numbers aren't consecutive, so we check the 2nd bit
                    //in pind register, but we write that value to the 3rd bit in the returned bitmask
                    value |= ((PIND & 2) ? 4 : 0);
                }
                if(PWMOn(3))
                {
                    /*digital pin 3 - PD0*/
                    /*PWM timer is on this pin, so check the disable bit*/
                    if(PWMDisable & 1)
                    {
                        if(!((pinConfigurations[0] & 8) && (DDRD & 1) ))
                        {
                            /*the pin is good to be read from*/
                            /*before we read from this pin, because it is a possible pwm pin, and may have previously been in */
                            /*a PWM state, we need to turn off PWM on this pin*/
                            turnOffPWM(digitalPinToTimer(3));
                            /*before reading, set pin as input, then read it and add the value into the value bitmask*/
                            /*set the mode as input*/
                            DDRD &= ~(1);
                            /*read the input into the value bitmask*/
                            value |= ((PIND & 1) ? 8 : 0);
                        }
                    }
                }
                else if(!((pinConfigurations[0] & 8) && (DDRD & 1) ))
                {
                    //pwm timer isn't on, use normal read operation
                    DDRD &= ~(1);
                    value |= ((PIND & 1) ? 8 : 0);
                }
                if(!((pinConfigurations[0] & 16) && (DDRD & 16) ))
                {
                    /*digital pin 4 - PD4*/
                    /*the pin is good to be read from, so we need to set it as input, then read it and add the*/
                    /*value into the value bitmask*/
                    /*set the mode as input*/
                    DDRD &= ~(16);
                    /*read the input into the value bitmask*/
                    value |= ((PIND & 16) ? 16 : 0);
                }
                if(PWMOn(5))
                {
                    /*digital pin 5 - PC6*/
                    /*PWM timer is on this pin, so check the disable bit*/
                    if(PWMDisable & 2)
                    {
                        if(!((pinConfigurations[0] & 32) && (DDRC & 64) ))
                        {
                            /*digital pin 5 (PWM pin)*/
                            /*the pin is good to be read from*/
                            /*before we read from this pin, because it is a possible pwm pin, and may have previously been in */
                            /*a PWM state, we need to turn off PWM on this pin*/
                            turnOffPWM(digitalPinToTimer(5));
                            /*before reading, set pin as input, then read it and add the value into the value bitmask*/
                            /*set the mode as input*/
                            DDRC &= ~(64);
                            /*read the input into the value bitmask*/
                            value |= ((PINC & 64) ? 32 : 0);
                        }
                    }
                }
                else if(!((pinConfigurations[0] & 32) && (DDRC & 64) ))
                {
                    //pwm timer isn't on, use normal read operation
                    DDRC &= ~(64);
                    value |= ((PINC & 64) ? 32 : 0);
                }
                if(PWMOn(6))
                {
                    /*digital pin 6 - PD7*/
                    /*PWM timer is on this pin, so check the disable bit*/
                    if(PWMDisable & 4)
                    {
                        if(!((pinConfigurations[0] & 64) && (DDRD & 128) ))
                        {
                            /*digital pin 6 (PWM pin)*/
                            /*the pin is good to be read from*/
                            /*before we read from this pin, because it is a possible pwm pin, and may have previously been in */
                            /*a PWM state, we need to turn off PWM on this pin*/
                            turnOffPWM(digitalPinToTimer(6));
                            /*before reading, set pin as input, then read it and add the value into the value bitmask*/
                            /*set the mode as input*/
                            DDRD &= ~(128);
                            /*read the input into the value bitmask*/
                            value |= ((PIND & 128) ? 64 : 0);
                        }
                    }
                }
                else if(!((pinConfigurations[0] & 64) && (DDRD & 128) ))
                {
                    //pwm timer isn't on, do normal read
                    DDRD &= ~(128);
                    value |= ((PIND & 128) ? 64 : 0);
                }
                if(!((pinConfigurations[0] & 128) && (DDRE & 64) ))
                {
                    /*digital pin 7 - PE6*/
                    /*the pin is good to be read from, so we need to set it as input, then read it and add the*/
                    /*value into the value bitmask*/
                    /*set the mode as input*/
                    DDRE &= ~(64);
                    /*read the input into the value bitmask*/
                    value |= ((PINE & 64) ? 128 : 0);
                }
                //end of port, send the value
                firmataDigitalReadPacketSend(port,value);
                return;
            case 1:
                if(!((pinConfigurations[1] & 1) && (DDRB & 16) ))
                {
                    /*digital pin 8 - PB4*/
                    /*the pin is good to be read from, so we need to set it as input, then read it and add the*/
                    /*value into the value bitmask*/
                    /*set the mode as input*/
                    DDRB &= ~(16);
                    /*read the input into the value bitmask*/
                    value |= ((PINB & 16) ? 1 : 0);
                }
                if(PWMOn(9))
                {
                    /*digital pin 9 - PB5*/
                    /*PWM timer is on this pin, so check the disable bit*/
                    if(PWMDisable & 1)
                    {
                        if(!((pinConfigurations[1] & 2) && (DDRB & 32) ))
                        {
                            /*digital pin 9 (PWM pin)*/
                            /*the pin is good to be read from*/
                            /*before we read from this pin, because it is a possible pwm pin, and may have previously been in */
                            /*a PWM state, we need to turn off PWM on this pin*/
                            turnOffPWM(digitalPinToTimer(9));
                            /*before reading, set pin as input, then read it and add the value into the value bitmask*/
                            /*set the mode as input*/
                            DDRB &= ~(32);
                            /*read the input into the value bitmask*/
                            value |= ((PINB & 32) ? 2 : 0);
                        }
                    }
                }
                else if(!((pinConfigurations[1] & 2) && (DDRB & 32) ))
                {
                    /*digital pin 9 (PWM pin)*/
                    /*the pin is good to be read from*/
                    /*note don't need to turn off PWM here*/
                    /*before reading, set pin as input, then read it and add the value into the value bitmask*/
                    /*set the mode as input*/
                    DDRB &= ~(32);
                    /*read the input into the value bitmask*/
                    value |= ((PINB & 32) ? 2 : 0);
                }
                if(PWMOn(10))
                {
                    /*digital pin 10 - PB6*/
                    /*PWM timer is on this pin, so check the disable bit*/
                    if(PWMDisable & 2)
                    {
                        if(!((pinConfigurations[1] & 4) && (DDRB & 64) ))
                        {
                            /*digital pin 10 (PWM pin)*/
                            /*the pin is good to be read from*/
                            /*before we read from this pin, because it is a possible pwm pin, and may have previously been in */
                            /*a PWM state, we need to turn off PWM on this pin*/
                            turnOffPWM(digitalPinToTimer(10));
                            /*before reading, set pin as input, then read it and add the value into the value bitmask*/
                            /*set the mode as input*/
                            DDRB &= ~(64);
                            /*read the input into the value bitmask*/
                            value |= ((PINB & 64) ? 4 : 0);
                        }
                    }
                }
                else if(!((pinConfigurations[1] & 4) && (DDRB & 64) ))
                {
                    /*digital pin 10 (PWM pin)*/
                    /*the pin is good to be read from*/
                    /*note don't need to turn off PWM here*/
                    /*before reading, set pin as input, then read it and add the value into the value bitmask*/
                    /*set the mode as input*/
                    DDRB &= ~(64);
                    /*read the input into the value bitmask*/
                    value |= ((PINB & 64) ? 4 : 0);
                }
                if(PWMOn(11))
                {
                    /*digital pin 11 - PB7*/
                    /*PWM timer is on this pin, so check the disable bit*/
                    if(PWMDisable & 4)
                    {
                        if(!((pinConfigurations[1] & 8) && (DDRB & 128) ))
                        {
                            /*digital pin 11 (PWM pin)*/
                            /*the pin is good to be read from*/
                            /*before we read from this pin, because it is a possible pwm pin, and may have previously been in */
                            /*a PWM state, we need to turn off PWM on this pin*/
                            turnOffPWM(digitalPinToTimer(11));
                            /*before reading, set pin as input, then read it and add the value into the value bitmask*/
                            /*set the mode as input*/
                            DDRB &= ~(128);
                            /*read the input into the value bitmask*/
                            value |= ((PINB & 128) ? 8 : 0);
                        }
                    }
                }
                else if(!((pinConfigurations[1] & 8) && (DDRB & 128) ))
                {
                    /*digital pin 11 (PWM pin)*/
                    /*the pin is good to be read from*/
                    /*note don't need to turn off PWM here*/
                    /*before reading, set pin as input, then read it and add the value into the value bitmask*/
                    /*set the mode as input*/
                    DDRB &= ~(128);
                    /*read the input into the value bitmask*/
                    value |= ((PINB & 128) ? 8 : 0);                  
                }
                if(!((pinConfigurations[1] & 16) && (DDRD & 64) ))
                {
                    /*digital pin 12 - PD6*/
                    /*the pin is good to be read from, so we need to set it as input, then read it and add the*/
                    /*value into the value bitmask*/
                    /*set the mode as input*/
                    DDRD &= ~(64);
                    /*read the input into the value bitmask*/
                    value |= ((PIND & 64) ? 16 : 0);
                }
                if(!((pinConfigurations[1] & 32) && (DDRC & 128) ))
                {
                    /*digital pin 13 - PC7*/
                    /*the pin is good to be read from, so we need to set it as input, then read it and add the*/
                    /*value into the value bitmask*/
                    /*set the mode as input*/
                    DDRC &= ~(128);
                    /*read the input into the value bitmask*/
                    value |= ((PINC & 128) ? 32 : 0);
                }
                firmataDigitalReadPacketSend(port,value);
                return;
            case 2:
                /*PORTC*/
                if(!((pinConfigurations[2] & 1) && (DDRF & 128) ))
                {
                    /*analog pin 0 - PF7*/
                    /*the pin is good to be read from, so we need to set it as input, then read it and add the*/
                    /*value into the value bitmask*/
                    /*set the mode as input*/
                    DDRF &= ~(128);
                    /*read the input into the value bitmask*/
                    value |= ((PINF & 128) ? 1 : 0); 
                }
                if(!((pinConfigurations[2] & 2) && (DDRF & 64) ))
                {
                    /*analog pin 1 - PF6*/
                    /*the pin is good to be read from, so we need to set it as input, then read it and add the*/
                    /*value into the value bitmask*/
                    /*set the mode as input*/
                    DDRF &= ~(64);
                    /*read the input into the value bitmask*/
                    value |= ((PINF & 64) ? 2 : 0);
                }
                if(!((pinConfigurations[2] & 4) && (DDRF & 32) ))
                {
                    /*analog pin 2 - PF5*/
                    /*the pin is good to be read from, so we need to set it as input, then read it and add the*/
                    /*value into the value bitmask*/
                    /*set the mode as input*/
                    DDRF &= ~(32);
                    /*read the input into the value bitmask*/
                    value |= ((PINF & 32) ? 4 : 0);
                }
                if(!((pinConfigurations[2] & 8) && (DDRF & 16) ))
                {
                    /*analog pin 3 - PF4*/
                    /*the pin is good to be read from, so we need to set it as input, then read it and add the*/
                    /*value into the value bitmask*/
                    /*set the mode as input*/
                    DDRF &= ~(16);
                    /*read the input into the value bitmask*/
                    value |= ((PINF & 16) ? 8 : 0);
                }
                if(!((pinConfigurations[2] & 16) && (DDRF & 2) ))
                {
                    /*analog pin 4 - PF1*/
                    /*the pin is good to be read from, so we need to set it as input, then read it and add the*/
                    /*value into the value bitmask*/
                    /*set the mode as input*/
                    DDRF &= ~(2);
                    /*read the input into the value bitmask*/
                    value |= ((PINF & 2) ? 16 : 0);
                }
                if(!((pinConfigurations[2] & 32) && (DDRF & 1) ))
                {
                    /*analog pin 5 - PF0*/
                    /*the pin is good to be read from, so we need to set it as input, then read it and add the*/
                    /*value into the value bitmask*/
                    /*set the mode as input*/
                    DDRF &= ~(1);
                    /*read the input into the value bitmask*/
                    value |= ((PINF & 1) ? 32 : 0);
                }
                /*there is no PC6 and PC7 (actually those pins are for the clock crystal)*/
                firmataDigitalReadPacketSend(port,value);
                return;
            default:
                /*wrong port, don't do anything, just return*/
                return;
        }
    }
#endif
}





/*TODO: optimize this function so that the lookup tables for the pwm timers don't have to be used*/
/*TODO: optimize this by removing the turnOffPWM function calls when it is redundant*/
void writeTaskRun(task * writeTask, byte * pinConfigurations)
{
#if defined (ARDUINO_AVR_UNO)
	/*need to check which kind of write we are doing*/
	if (getType(writeTask) == ANALOG_WRITE)
	{
		/*analog write uses a pin*/
		byte pin = getPinNumber(writeTask);
		
		if ((pin == 3) || (pin ==5) || (pin==6) || (pin == 9) || (pin == 10) || (pin == 11))
		{
			pinMode(pin,OUTPUT);
			/*write the value to the pin, then exit*/
			analogWrite(pin,getValue(writeTask));
			return;
		}
		/*if it's not a PWM pin, just return without doing anything*/
		return;
	}
	else
	{
		/*the PWMDisable represents whether or not the pwm pins in this port are to be overwritten with the value from the actual value bitmask sent over the serial wire*/
		register byte PWMDisable = getPinNumber(writeTask);
		/*digital write uses a port, for the actual layout of the*/
		/*board, the following is the mapping between the */
		/*ATMEGA328P chip's ports and Firmata's port numbers*/
		/*port 0 is PORTD*/
		/*port 1 is PORTB*/
		/*port 2 is PORTC (analog pins)*/
		/*store the port in a register for fast writing*/
		register byte port = getPortNumber(writeTask);
		/*store this variable in a register, as it will be used for the write operation, which has been optimized*/
		/*in the following to allow a write task to happen as fast as possible*/
		register byte value = getValue(writeTask);
		/*for each pin in the port, need to check if the pin was hard configured to be whatever state it is in now*/
		/*we can check the state it is in now by checking the register DDRX, where X is the port (so B, C, or D)*/
		/*the register DDRX holds the states of all of the pins in the port, so we have to only check the pin we are*/
		/*concerned with*/
		/*we can check whether or not the pin was hard configured to its state with the array of bytes, pinConfigurations*/
		/*where each byte has the configuration for each pin as a bitmask, 1 is hard configured, 0 is soft configured*/
		/*the distinction is important, because if a user has not said to configure a pin a particular direction,*/
		/*the server needs to accomodate reading or writing, so it will switch the pinMode at will*/
		/*if however the user did configure a pin to have a particular direction, then we cannot change that direction*/
		/*without first receiving a command from the user to do so, for example if the user says to configure a pin as */
		/*input, then sends a write command, we must ignore that command. If the user instead issued a write command, then*/
		/*a read command, we would switch to an input after receiving the read command because the user did not explicitly*/
		/*say to configure the pin a particular way*/
		/*the english logic is that we should only disallow the write if the pin is hard configured as an input*/
		/*the boolean logic is as follows:*/
		/* data direction bit | hard configured? | allow write?*/
		/*        0           |        0         |      1*/
		/*        0           |        1         |      0*/
		/*        1           |        0         |      1*/
		/*        1           |        1         |      1*/
		/*this yields the following boolean expression: allow write = OR( NOT( hard Configured? ) , data direction bit )*/
		/*we do this for each pin in the port, but need to switch on the port*/
		/*keeping this as non looping code helps the compiler optimize the code*/
		switch(port)
		{
			
			/*we bitwise and this with each byte for the port, it doesn't make a difference that it may not be 1, as long as*/
			/*it is not zero, it will be considered as a 1*/
			/*note that the bitwise and is used, while the logical or and logical not are used upon the result of the*/
			/*bitwise and*/
			case 0:
				/*PORTD (note we skip pins 0 and 1 as they are reserved for SerialLink communication)*/
				if(!(pinConfigurations[0] & 4) || (DDRD & 4) )
				{
					/*digital pin 2*/
					/*we are good to configure the pin for writing, and write it's value*/
					DDRD |= 4;
					/*for safety turn off interrupts for the write operation*/
					noInterrupts();
					if (value & 4)
					{
						/*writes a value of 1 to the 2nd bit*/
						PORTD |= 4;
					}
					else
					{
						/*writes a value of 0 to the 2nd bit*/
						PORTD &= ~4;
					}
					interrupts();
				}
				/*because this is a pwm pin, check if the pwm timer is on, if it is then we have to check the hidden bits*/
				if(PWMOn(3))
				{
					/*it is on, so we should check the hidden bits, for this pwm pin, it will be the first bit*/
					if(PWMDisable & 1)
					{
						/*pwm disabling is on, so turn off pwm to set this pin's value*/
						turnOffPWM(digitalPinToTimer(3));
						noInterrupts();
						if (value & 8)
						{
							/*writes a value of 1 to the 3rd bit*/
							PORTD |= 8;
						}
						else
						{
							/*writes a value of 0 to the 3rd bit*/
							PORTD &= ~8;
						}
						interrupts();
					}
					/*no else, as the user has requested that we not turn off the pwm, so don't bother this pin*/
				}
				//pwm isn't on, so do a normal check
				else if(!(pinConfigurations[0] & 8) || (DDRD & 8))
				{
					/*digital pin 3 (PWM pin)*/
					/*we are good to configure the pin for writing, and write it's value*/
					DDRD |= 8;
					/*for safety turn off interrupts for the write operation*/
					noInterrupts();
					if (value & 8)
					{
						/*writes a value of 1 to the 3rd bit*/
						PORTD |= 8;
					}
					else
					{
						/*writes a value of 0 to the 3rd bit*/
						PORTD &= ~8;
					}
					interrupts();
				}
				if(!(pinConfigurations[0] & 16) || (DDRD & 16))
				{
					/*digital pin 4*/
					/*we are good to configure the pin for writing, and write it's value*/
					DDRD |= 16;
					/*for safety turn off interrupts for the write operation*/
					noInterrupts();
					if (value & 16)
					{
						/*writes a value of 1 to the 4th bit*/
						PORTD |= 16;
					}
					else
					{
						/*writes a value of 0 to the 4th bit*/
						PORTD &= ~16;
					}
					interrupts();
				}
				/*because this is a pwm pin, check if the pwm timer is on, if it is then we have to check the hidden bits*/
				if(PWMOn(5))
				{
					/*it is on, so we should check the hidden bits, for this pwm pin, it will be the second bit*/
					if(PWMDisable & 2)
					{
						/*pwm disabling is on, so turn off pwm to set this pin's value*/
						turnOffPWM(digitalPinToTimer(5));
						noInterrupts();
						if (value & 32)
						{
							/*writes a value of 1 to the 3rd bit*/
							PORTD |= 32;
						}
						else
						{
							/*writes a value of 0 to the 3rd bit*/
							PORTD &= ~32;
						}
						interrupts();
					}
					/*no else, as the user has requested that we not turn off the pwm, so don't bother this pin*/
				}
				/*pwm isn't on, so do a normal check*/
				else if(!(pinConfigurations[0] & 32) || (DDRD & 32))
				{
					/*digital pin 5 (PWM pin)*/
					/*we are good to configure the pin for writing, and write it's value*/
					DDRD |= 32;
					/*before we write to this pin, we need to turn off pwm timer on this pin*/
					turnOffPWM(digitalPinToTimer(5));
					/*for safety turn off interrupts for the write operation*/
					noInterrupts();
					if (value & 32)
					{
						/*writes a value of 1 to the 3rd bit*/
						PORTD |= 32;
					}
					else
					{
						/*writes a value of 0 to the 3rd bit*/
						PORTD &= ~32;
					}
					interrupts();
				}
				/*because this is a pwm pin, check if the pwm timer is on, if it is then we have to check the hidden bits*/
				if(PWMOn(6))
				{
					/*it is on, so we should check the hidden bits, for this pwm pin, it will be the third bit*/
					if(PWMDisable & 4)
					{
						/*pwm disabling is on, so turn off pwm to set this pin's value*/
						turnOffPWM(digitalPinToTimer(6));
						noInterrupts();
						if (value & 64)
						{
							/*writes a value of 1 to the 3rd bit*/
							PORTD |= 64;
						}
						else
						{
							/*writes a value of 0 to the 3rd bit*/
							PORTD &= ~64;
						}
						interrupts();
					}
					/*no else, as the user has requested that we not turn off the pwm, so don't bother this pin*/
				}
				/*pwm isn't on, so do a normal check*/
				else if(!(pinConfigurations[0] & 64) || (DDRD & 64))
				{
					/*digital pin 6 (PWM pin)*/
					/*we are good to configure the pin for writing, and write it's value*/
					DDRD |= 64;
					/*before we write to this pin, we need to turn off pwm timer on this pin*/
					turnOffPWM(digitalPinToTimer(6));
					/*for safety turn off interrupts for the write operation*/
					noInterrupts();
					if (value & 64)
					{
						/*writes a value of 1 to the 3rd bit*/
						PORTD |= 64;
					}
					else
					{
						/*writes a value of 0 to the 3rd bit*/
						PORTD &= ~64;
					}
					interrupts();
				}
				if(!(pinConfigurations[0] & 128) || (DDRD & 128))
				{
					/*digital pin 7*/
					/*we are good to configure the pin for writing, and write it's value*/
					DDRD |= 128;
					/*for safety turn off interrupts for the write operation*/
					noInterrupts();
					if (value & 128)
					{
						/*writes a value of 1 to the 7th bit*/
						PORTD |= 128;
					}
					else
					{
						/*writes a value of 0 to the 7th bit*/
						PORTD &= ~128;
					}
					interrupts();
				}
				/*done writing, we can return*/
				return;
			case 1:
				/*PORTB*/
				if(!(pinConfigurations[1] & 1) || (DDRB & 1))
				{
					/*digital pin 8*/
					/*we are good to configure the pin for writing, and write it's value*/
					DDRB |= 1;
					/*for safety turn off interrupts for the write operation*/
					noInterrupts();
					if (value & 1)
					{
						/*writes a value of 1 to the 0th bit*/
						PORTB |= 1;
					}
					else
					{
						/*writes a value of 0 to the 0th bit*/
						PORTB &= ~1;
					}
					interrupts();
				}
				if(!(pinConfigurations[1] & 2) || (DDRB & 2))
				{
					/*digital pin 9 (PWM pin)*/
					/*we are good to configure the pin for writing, and write it's value*/
					DDRB |= 2;
					/*before we write to this pin, because it is a possible pwm pin, and may have previously been in */
					/*a PWM state, we need to turn off PWM on this pin*/
					turnOffPWM(digitalPinToTimer(9));
					/*for safety turn off interrupts for the write operation*/
					noInterrupts();
					if (value & 2)
					{
						/*writes a value of 1 to the 1st bit*/
						PORTD |= 2;
					}
					else
					{
						/*writes a value of 0 to the 1st bit*/
						PORTD &= ~2;
					}
					interrupts();
				}
				/*because this is a pwm pin, check if the pwm timer is on, if it is then we have to check the hidden bits*/
				if(PWMOn(9))
				{
					/*it is on, so we should check the hidden bits, for this pwm pin, it will be the first bit*/
					if(PWMDisable & 1)
					{
						/*pwm disabling is on, so turn off pwm to set this pin's value*/
						turnOffPWM(digitalPinToTimer(9));
						noInterrupts();
						if (value & 2)
						{
							/*writes a value of 1 to the 3rd bit*/
							PORTD |= 2;
						}
						else
						{
							/*writes a value of 0 to the 3rd bit*/
							PORTD &= ~2;
						}
						interrupts();
					}
					/*no else, as the user has requested that we not turn off the pwm, so don't bother this pin*/
				}
				/*pwm isn't on, so do a normal check*/
				else if(!(pinConfigurations[1] & 2) || (DDRB & 2))
				{
					/*digital pin 9 (PWM pin)*/
					/*we are good to configure the pin for writing, and write it's value*/
					DDRB |= 2;
					/*before we write to this pin, we need to turn off pwm timer on this pin*/
					turnOffPWM(digitalPinToTimer(9));
					/*for safety turn off interrupts for the write operation*/
					noInterrupts();
					if (value & 2)
					{
						/*writes a value of 1 to the 3rd bit*/
						PORTB |= 2;
					}
					else
					{
						/*writes a value of 0 to the 3rd bit*/
						PORTB &= ~2;
					}
					interrupts();
				}
				if(!(pinConfigurations[1] & 4) || (DDRB & 4))
				{
					/*digital pin 10 (PWM pin)*/
					/*we are good to configure the pin for writing, and write it's value*/
					DDRB |= 4;
					/*before we write to this pin, because it is a possible pwm pin, and may have previously been in */
					/*a PWM state, we need to turn off PWM on this pin*/
					turnOffPWM(digitalPinToTimer(10));
					/*for safety turn off interrupts for the write operation*/
					noInterrupts();
					if (value & 4)
					{
						/*writes a value of 1 to the 2nd bit*/
						PORTB |= 4;
					}
					else
					{
						/*writes a value of 0 to the 2nd bit*/
						PORTB &= ~4;
					}
					interrupts();
				}
				/*because this is a pwm pin, check if the pwm timer is on, if it is then we have to check the hidden bits*/
				if(PWMOn(10))
				{
					/*digital pin 10 (PWM pin)*/
					/*it is on, so we should check the hidden bits, for this pwm pin, it will be the second bit*/
					if(PWMDisable & 2)
					{
						/*pwm disabling is on, so turn off pwm to set this pin's value*/
						turnOffPWM(digitalPinToTimer(10));
						noInterrupts();
						if (value & 4)
						{
							/*writes a value of 1 to the 3rd bit*/
							PORTB |= 4;
						}
						else
						{
							/*writes a value of 0 to the 3rd bit*/
							PORTB &= ~4;
						}
						interrupts();
					}
					/*no else, as the user has requested that we not turn off the pwm, so don't bother this pin*/
				}
				/*pwm isn't on, so do a normal check*/
				else if(!(pinConfigurations[1] & 4) || (DDRD & 4))
				{
					/*digital pin 10 (PWM pin)*/
					/*we are good to configure the pin for writing, and write it's value*/
					DDRB |= 4;
					/*before we write to this pin, we need to turn off pwm timer on this pin*/
					turnOffPWM(digitalPinToTimer(10));
					/*for safety turn off interrupts for the write operation*/
					noInterrupts();
					if (value & 4)
					{
						/*writes a value of 1 to the 3rd bit*/
						PORTB |= 4;
					}
					else
					{
						/*writes a value of 0 to the 3rd bit*/
						PORTB &= ~4;
					}
					interrupts();
				}
				/*because this is a pwm pin, check if the pwm timer is on, if it is then we have to check the hidden bits*/
				if(PWMOn(11))
				{
					/*digital pin 11 (PWM pin)*/
					/*it is on, so we should check the hidden bits, for this pwm pin, it will be the third bit*/
					if(PWMDisable & 4)
					{
						/*pwm disabling is on, so turn off pwm to set this pin's value*/
						turnOffPWM(digitalPinToTimer(11));
						noInterrupts();
						if (value & 8)
						{
							/*writes a value of 1 to the 3rd bit*/
							PORTB |= 8;
						}
						else
						{
							/*writes a value of 0 to the 3rd bit*/
							PORTB &= ~8;
						}
						interrupts();
					}
					/*no else, as the user has requested that we not turn off the pwm, so don't bother this pin*/
				}
				/*pwm isn't on, so do a normal check*/
				else if(!(pinConfigurations[1] & 8) || (DDRD & 8))
				{
					/*digital pin 11 (PWM pin)*/
					/*we are good to configure the pin for writing, and write it's value*/
					DDRB |= 8;
					/*before we write to this pin, we need to turn off pwm timer on this pin*/
					turnOffPWM(digitalPinToTimer(11));
					/*for safety turn off interrupts for the write operation*/
					noInterrupts();
					if (value & 8)
					{
						/*writes a value of 1 to the 3rd bit*/
						PORTB |= 8;
					}
					else
					{
						/*writes a value of 0 to the 3rd bit*/
						PORTB &= ~8;
					}
					interrupts();
				}
				if(!(pinConfigurations[1] & 16) || (DDRB & 16))
				{
					/*digital pin 12*/
					/*we are good to configure the pin for writing, and write it's value*/
					DDRB |= 16;
					/*for safety turn off interrupts for the write operation*/
					noInterrupts();
					if (value & 16)
					{
						/*writes a value of 1 to the 4th bit*/
						PORTB |= 16;
					}
					else
					{
						/*writes a value of 0 to the 4th bit*/
						PORTB &= ~16;
					}
					interrupts();
				}
				if(!(pinConfigurations[1] & 32) || (DDRB & 32))
				{
					/*digital pin 13*/
					/*we are good to configure the pin for writing, and write it's value*/
					DDRB |= 32;
					/*for saferty turn interrupts off*/
					/*for safety turn off interrupts for the write operation*/
					noInterrupts();
					if (value & 32)
					{
						/*writes a value of 1 to the 5th bit*/
						PORTB |= 32;
					}
					else
					{
						/*writes a value of 0 to the 5th bit*/
						PORTB &= ~32;
					}
					interrupts();
				}
				/*there is no PB6 and PB7 (actually they're used by the clock crystal)*/
				/*done writing we can return*/
				return;
			case 2:
				/*PORTC*/
				if(!(pinConfigurations[2] & 1) || (DDRC & 1))
				{
					/*analog pin 0*/
					/*we are good to configure the pin for writing, and write it's value*/
					DDRC |= 1;
					/*for safety turn off interrupts for the write operation*/
					noInterrupts();
					if (value & 1)
					{
						/*writes a value of 1 to the 0th bit*/
						PORTC |= 1;
					}
					else
					{
						/*writes a value of 0 to the 0th bit*/
						PORTC &= ~1;
					}
					interrupts();
				}
				if(!(pinConfigurations[2] & 2) || (DDRC & 2))
				{
					/*analog pin 1*/
					/*we are good to configure the pin for writing, and write it's value*/
					DDRC |= 2;
					/*for safety turn off interrupts for the write operation*/
					noInterrupts();
					if (value & 2)
					{
						/*writes a value of 1 to the 1st bit*/
						PORTC |= 2;
					}
					else
					{
						/*writes a value of 0 to the 1st bit*/
						PORTC &= ~2;
					}
					interrupts();
				}
				if(!(pinConfigurations[2] & 4) || (DDRC & 4))
				{
					/*analog pin 2*/
					/*we are good to configure the pin for writing, and write it's value*/
					DDRC |= 4;
					/*for safety turn off interrupts for the write operation*/
					noInterrupts();
					if (value & 4)
					{
						/*writes a value of 1 to the 2nd bit*/
						PORTC |= 4;
					}
					else
					{
						/*writes a value of 0 to the 2nd bit*/
						PORTC &= ~4;
					}
					interrupts();
				}
				if(!(pinConfigurations[2] & 8) || (DDRC & 8))
				{
					/*analog pin 3*/
					/*we are good to configure the pin for writing, and write it's value*/
					DDRC |= 8;
					/*for safety turn off interrupts for the write operation*/
					noInterrupts();
					if (value & 8)
					{
						/*writes a value of 1 to the 3rd bit*/
						PORTC |= 8;
					}
					else
					{
						/*writes a value of 0 to the 3rd bit*/
						PORTC &= ~8;
					}
					interrupts();
				}
				if(!(pinConfigurations[2] & 16) || (DDRC & 16))
				{
					/*analog pin 4*/
					/*we are good to configure the pin for writing, and write it's value*/
					DDRC |= 16;
					/*for safety turn off interrupts for the write operation*/
					noInterrupts();
					if (value & 16)
					{
						/*writes a value of 1 to the 4th bit*/
						PORTC |= 16;
					}
					else
					{
						/*writes a value of 0 to the 4th bit*/
						PORTC &= ~16;
					}
					interrupts();
				}
				if(!(pinConfigurations[2] & 32) || (DDRC & 32))
				{
					/*analog pin 5*/
					/*we are good to configure the pin for writing, and write it's value*/
					DDRC |= 32;
					/*for safety turn off interrupts for the write operation*/
					noInterrupts();
					if (value & 32)
					{
						/*writes a value of 1 to the 5th bit*/
						PORTC |= 32;
					}
					else
					{
						/*writes a value of 0 to the 5th bit*/
						PORTC &= ~32;
					}
					interrupts();
				}
				/*there is not PC6 (actually it's the reset pin) and PC7*/
				/*done writing we can return*/
				return;
			default:
				/*wrong port, just return without doing anything*/
				return;
		}
	}
#elif defined (ARDUINO_AVR_YUN)
    /*need to check which kind of write we are doing*/
    if (getType(writeTask) == ANALOG_WRITE)
    {
        /*analog write uses a pin*/
        byte pin = getPinNumber(writeTask);
        
        //arduino yun has pwm on pin 13 as well
        if ((pin == 3) || (pin ==5) || (pin==6) || (pin == 9) || (pin == 10) || (pin == 11) || pin == 13)
        {
            pinMode(pin,OUTPUT);
            /*write the value to the pin, then exit*/
            analogWrite(pin,getValue(writeTask));
            return;
        }
        /*if it's not a PWM pin, just return without doing anything*/
        return;
    }
    else
    {
        /*the PWMDisable represents whether or not the pwm pins in this port are to be overwritten with the value from the actual value bitmask sent over the serial wire*/
        register byte PWMDisable = getPinNumber(writeTask);
        /*digital write uses a port, for the actual layout of the*/
        /*board, the following is the mapping between the */
        /*ATMEGA328P chip's ports and Firmata's port numbers*/
        /*port 0 is PORTD*/
        /*port 1 is PORTB*/
        /*port 2 is PORTC (analog pins)*/
        /*store the port in a register for fast writing*/
        register byte port = getPortNumber(writeTask);
        /*store this variable in a register, as it will be used for the write operation, which has been optimized*/
        /*in the following to allow a write task to happen as fast as possible*/
        register byte value = getValue(writeTask);
        /*for each pin in the port, need to check if the pin was hard configured to be whatever state it is in now*/
        /*we can check the state it is in now by checking the register DDRX, where X is the port (so B, C, or D)*/
        /*the register DDRX holds the states of all of the pins in the port, so we have to only check the pin we are*/
        /*concerned with*/
        /*we can check whether or not the pin was hard configured to its state with the array of bytes, pinConfigurations*/
        /*where each byte has the configuration for each pin as a bitmask, 1 is hard configured, 0 is soft configured*/
        /*the distinction is important, because if a user has not said to configure a pin a particular direction,*/
        /*the server needs to accomodate reading or writing, so it will switch the pinMode at will*/
        /*if however the user did configure a pin to have a particular direction, then we cannot change that direction*/
        /*without first receiving a command from the user to do so, for example if the user says to configure a pin as */
        /*input, then sends a write command, we must ignore that command. If the user instead issued a write command, then*/
        /*a read command, we would switch to an input after receiving the read command because the user did not explicitly*/
        /*say to configure the pin a particular way*/
        /*the english logic is that we should only disallow the write if the pin is hard configured as an input*/
        /*the boolean logic is as follows:*/
        /* data direction bit | hard configured? | allow write?*/
        /*        0           |        0         |      1*/
        /*        0           |        1         |      0*/
        /*        1           |        0         |      1*/
        /*        1           |        1         |      1*/
        /*this yields the following boolean expression: allow write = OR( NOT( hard Configured? ) , data direction bit )*/
        /*we do this for each pin in the port, but need to switch on the port*/
        /*keeping this as non looping code helps the compiler optimize the code*/
        switch(port)
        {
            
            /*we bitwise and this with each byte for the port, it doesn't make a difference that it may not be 1, as long as*/
            /*it is not zero, it will be considered as a 1*/
            /*note that the bitwise and is used, while the logical or and logical not are used upon the result of the*/
            /*bitwise and*/
            case 0:
                /*PORTD (note we skip pins 0 and 1 as they are reserved for SerialLink communication)*/
                if(!(pinConfigurations[0] & 4) || (DDRD & 2) )
                {
                    /*digital pin 2 - PD1*/
                    /*we are good to configure the pin for writing, and write it's value*/
                    DDRD |= 2;
                    /*for safety turn off interrupts for the write operation*/
                    noInterrupts();
                    if (value & 4)
                    {
                        /*writes a value of 1 to the 2nd bit*/
                        PORTD |= 2;
                    }
                    else
                    {
                        /*writes a value of 0 to the 2nd bit*/
                        PORTD &= ~2;
                    }
                    interrupts();
                }
                /*because this is a pwm pin, check if the pwm timer is on, if it is then we have to check the hidden bits*/
                if(PWMOn(3))
                {
                    /*digital pin 3 (PWM pin) - PD0*/
                    /*it is on, so we should check the hidden bits, for this pwm pin, it will be the first bit*/
                    if(PWMDisable & 1)
                    {
                        /*pwm disabling is on, so turn off pwm to set this pin's value*/
                        turnOffPWM(digitalPinToTimer(3));
                        noInterrupts();
                        if (value & 8)
                        {
                            /*writes a value of 1 to the 3rd bit*/
                            PORTD |= 1;
                        }
                        else
                        {
                            /*writes a value of 0 to the 3rd bit*/
                            PORTD &= ~1;
                        }
                        interrupts();
                    }
                    /*no else, as the user has requested that we not turn off the pwm, so don't bother this pin*/
                }
                //pwm isn't on, so do a normal check
                else if(!(pinConfigurations[0] & 8) || (DDRD & 1))
                {
                    /*digital pin 3 (PWM pin) - PD0*/
                    /*we are good to configure the pin for writing, and write it's value*/
                    DDRD |= 1;
                    /*for safety turn off interrupts for the write operation*/
                    noInterrupts();
                    if (value & 8)
                    {
                        /*writes a value of 1 to the 3rd bit*/
                        PORTD |= 1;
                    }
                    else
                    {
                        /*writes a value of 0 to the 3rd bit*/
                        PORTD &= ~1;
                    }
                    interrupts();
                }
                if(!(pinConfigurations[0] & 16) || (DDRD & 16))
                {
                    /*digital pin 4 - PD4*/
                    /*we are good to configure the pin for writing, and write it's value*/
                    DDRD |= 16;
                    /*for safety turn off interrupts for the write operation*/
                    noInterrupts();
                    if (value & 16)
                    {
                        /*writes a value of 1 to the 4th bit*/
                        PORTD |= 16;
                    }
                    else
                    {
                        /*writes a value of 0 to the 4th bit*/
                        PORTD &= ~16;
                    }
                    interrupts();
                }
                /*because this is a pwm pin, check if the pwm timer is on, if it is then we have to check the hidden bits*/
                if(PWMOn(5))
                {
                    /*digital pin 5 (PWM pin) - PC6*/
                    /*it is on, so we should check the hidden bits, for this pwm pin, it will be the second bit*/
                    if(PWMDisable & 2)
                    {
                        /*pwm disabling is on, so turn off pwm to set this pin's value*/
                        turnOffPWM(digitalPinToTimer(5));
                        noInterrupts();
                        if (value & 32)
                        {
                            /*writes a value of 1 to the 3rd bit*/
                            PORTC |= 64;
                        }
                        else
                        {
                            /*writes a value of 0 to the 3rd bit*/
                            PORTC &= ~64;
                        }
                        interrupts();
                    }
                    /*no else, as the user has requested that we not turn off the pwm, so don't bother this pin*/
                }
                /*pwm isn't on, so do a normal check*/
                else if(!(pinConfigurations[0] & 32) || (DDRC & 64))
                {
                    /*digital pin 5 (PWM pin) - PC6*/
                    /*we are good to configure the pin for writing, and write it's value*/
                    DDRC |= 32;
                    /*before we write to this pin, we need to turn off pwm timer on this pin*/
                    turnOffPWM(digitalPinToTimer(5));
                    /*for safety turn off interrupts for the write operation*/
                    noInterrupts();
                    if (value & 32)
                    {
                        /*writes a value of 1 to the 3rd bit*/
                        PORTC |= 64;
                    }
                    else
                    {
                        /*writes a value of 0 to the 3rd bit*/
                        PORTC &= ~64;
                    }
                    interrupts();
                }
                /*because this is a pwm pin, check if the pwm timer is on, if it is then we have to check the hidden bits*/
                if(PWMOn(6))
                {
                    /*digital pin 6 (PWM pin) - PD7*/
                    /*it is on, so we should check the hidden bits, for this pwm pin, it will be the third bit*/
                    if(PWMDisable & 4)
                    {
                        /*pwm disabling is on, so turn off pwm to set this pin's value*/
                        turnOffPWM(digitalPinToTimer(6));
                        noInterrupts();
                        if (value & 64)
                        {
                            /*writes a value of 1 to the 3rd bit*/
                            PORTD |= 128;
                        }
                        else
                        {
                            /*writes a value of 0 to the 3rd bit*/
                            PORTD &= ~128;
                        }
                        interrupts();
                    }
                    /*no else, as the user has requested that we not turn off the pwm, so don't bother this pin*/
                }
                /*pwm isn't on, so do a normal check*/
                else if(!(pinConfigurations[0] & 64) || (DDRD & 128))
                {
                    /*digital pin 6 (PWM pin) - PD7*/
                    /*we are good to configure the pin for writing, and write it's value*/
                    DDRD |= 64;
                    /*before we write to this pin, we need to turn off pwm timer on this pin*/
                    turnOffPWM(digitalPinToTimer(6));
                    /*for safety turn off interrupts for the write operation*/
                    noInterrupts();
                    if (value & 64)
                    {
                        /*writes a value of 1 to the 3rd bit*/
                        PORTD |= 128;
                    }
                    else
                    {
                        /*writes a value of 0 to the 3rd bit*/
                        PORTD &= ~128;
                    }
                    interrupts();
                }
                if(!(pinConfigurations[0] & 128) || (DDRE & 64))
                {
                    /*digital pin 7 - PE6*/
                    /*we are good to configure the pin for writing, and write it's value*/
                    DDRE |= 64;
                    /*for safety turn off interrupts for the write operation*/
                    noInterrupts();
                    if (value & 128)
                    {
                        /*writes a value of 1 to the 7th bit*/
                        PORTE |= 64;
                    }
                    else
                    {
                        /*writes a value of 0 to the 7th bit*/
                        PORTE &= ~64;
                    }
                    interrupts();
                }
                /*done writing, we can return*/
                return;
            case 1:
                /*PORTB*/
                if(!(pinConfigurations[1] & 1) || (DDRB & 16))
                {
                    /*digital pin 8 - PB4*/
                    /*we are good to configure the pin for writing, and write it's value*/
                    DDRB |= 16;
                    /*for safety turn off interrupts for the write operation*/
                    noInterrupts();
                    if (value & 1)
                    {
                        /*writes a value of 1 to the 0th bit*/
                        PORTB |= 16;
                    }
                    else
                    {
                        /*writes a value of 0 to the 0th bit*/
                        PORTB &= ~16;
                    }
                    interrupts();
                }
                /*because this is a pwm pin, check if the pwm timer is on, if it is then we have to check the hidden bits*/
                if(PWMOn(9))
                {
                    /*digital pin 9 (PWM pin) - PB5*/
                    /*it is on, so we should check the hidden bits, for this pwm pin, it will be the first bit*/
                    if(PWMDisable & 1)
                    {
                        /*pwm disabling is on, so turn off pwm to set this pin's value*/
                        turnOffPWM(digitalPinToTimer(9));
                        noInterrupts();
                        if (value & 2)
                        {
                            /*writes a value of 1 to the 3rd bit*/
                            PORTB |= 32;
                        }
                        else
                        {
                            /*writes a value of 0 to the 3rd bit*/
                            PORTB &= ~32;
                        }
                        interrupts();
                    }
                    /*no else, as the user has requested that we not turn off the pwm, so don't bother this pin*/
                }
                /*pwm isn't on, so do a normal check*/
                else if(!(pinConfigurations[1] & 2) || (DDRB & 32))
                {
                    /*digital pin 9 (PWM pin) - PB5*/
                    /*we are good to configure the pin for writing, and write it's value*/
                    DDRB |= 32;
                    /*before we write to this pin, we need to turn off pwm timer on this pin*/
                    turnOffPWM(digitalPinToTimer(9));
                    /*for safety turn off interrupts for the write operation*/
                    noInterrupts();
                    if (value & 2)
                    {
                        /*writes a value of 1 to the 3rd bit*/
                        PORTB |= 32;
                    }
                    else
                    {
                        /*writes a value of 0 to the 3rd bit*/
                        PORTB &= ~32;
                    }
                    interrupts();
                }
                /*because this is a pwm pin, check if the pwm timer is on, if it is then we have to check the hidden bits*/
                if(PWMOn(10))
                {
                    /*digital pin 10 (PWM pin) - PB6*/
                    /*it is on, so we should check the hidden bits, for this pwm pin, it will be the second bit*/
                    if(PWMDisable & 2)
                    {
                        /*pwm disabling is on, so turn off pwm to set this pin's value*/
                        turnOffPWM(digitalPinToTimer(10));
                        noInterrupts();
                        if (value & 4)
                        {
                            /*writes a value of 1 to the 3rd bit*/
                            PORTB |= 64;
                        }
                        else
                        {
                            /*writes a value of 0 to the 3rd bit*/
                            PORTB &= ~64;
                        }
                        interrupts();
                    }
                    /*no else, as the user has requested that we not turn off the pwm, so don't bother this pin*/
                }
                /*pwm isn't on, so do a normal check*/
                else if(!(pinConfigurations[1] & 4) || (DDRB & 64))
                {
                    /*digital pin 10 (PWM pin) - PB6*/
                    /*we are good to configure the pin for writing, and write it's value*/
                    DDRB |= 64;
                    /*before we write to this pin, we need to turn off pwm timer on this pin*/
                    turnOffPWM(digitalPinToTimer(10));
                    /*for safety turn off interrupts for the write operation*/
                    noInterrupts();
                    if (value & 4)
                    {
                        /*writes a value of 1 to the 3rd bit*/
                        PORTB |= 64;
                    }
                    else
                    {
                        /*writes a value of 0 to the 3rd bit*/
                        PORTB &= ~64;
                    }
                    interrupts();
                }
                /*because this is a pwm pin, check if the pwm timer is on, if it is then we have to check the hidden bits*/
                if(PWMOn(11))
                {
                    /*digital pin 11 (PWM pin) - PB7*/
                    /*it is on, so we should check the hidden bits, for this pwm pin, it will be the third bit*/
                    if(PWMDisable & 4)
                    {
                        /*pwm disabling is on, so turn off pwm to set this pin's value*/
                        turnOffPWM(digitalPinToTimer(11));
                        noInterrupts();
                        if (value & 8)
                        {
                            /*writes a value of 1 to the 3rd bit*/
                            PORTB |= 128;
                        }
                        else
                        {
                            /*writes a value of 0 to the 3rd bit*/
                            PORTB &= ~128;
                        }
                        interrupts();
                    }
                    /*no else, as the user has requested that we not turn off the pwm, so don't bother this pin*/
                }
                /*pwm isn't on, so do a normal check*/
                else if(!(pinConfigurations[1] & 8) || (DDRD & 128))
                {
                    /*digital pin 11 (PWM pin) - PB7*/
                    /*we are good to configure the pin for writing, and write it's value*/
                    DDRB |= 128;
                    /*before we write to this pin, we need to turn off pwm timer on this pin*/
                    turnOffPWM(digitalPinToTimer(11));
                    /*for safety turn off interrupts for the write operation*/
                    noInterrupts();
                    if (value & 8)
                    {
                        /*writes a value of 1 to the 3rd bit*/
                        PORTB |= 128;
                    }
                    else
                    {
                        /*writes a value of 0 to the 3rd bit*/
                        PORTB &= ~128;
                    }
                    interrupts();
                }
                if(!(pinConfigurations[1] & 16) || (DDRD & 64))
                {
                    /*digital pin 12 - PD6*/
                    /*we are good to configure the pin for writing, and write it's value*/
                    DDRD |= 64;
                    /*for safety turn off interrupts for the write operation*/
                    noInterrupts();
                    if (value & 16)
                    {
                        /*writes a value of 1 to the 4th bit*/
                        PORTD |= 64;
                    }
                    else
                    {
                        /*writes a value of 0 to the 4th bit*/
                        PORTD &= ~64;
                    }
                    interrupts();
                }
                if(!(pinConfigurations[1] & 32) || (DDRC & 128))
                {
                    /*digital pin 13 - PC7*/
                    /*we are good to configure the pin for writing, and write it's value*/
                    DDRC |= 128;
                    /*for saferty turn interrupts off*/
                    /*for safety turn off interrupts for the write operation*/
                    noInterrupts();
                    if (value & 32)
                    {
                        /*writes a value of 1 to the 5th bit*/
                        PORTC |= 128;
                    }
                    else
                    {
                        /*writes a value of 0 to the 5th bit*/
                        PORTC &= ~128;
                    }
                    interrupts();
                }
                /*done writing we can return*/
                return;
            case 2:
                if(!(pinConfigurations[2] & 1) || (DDRF & 128))
                {
                    /*analog pin 0 - PF7*/
                    /*we are good to configure the pin for writing, and write it's value*/
                    DDRF |= 128;
                    /*for safety turn off interrupts for the write operation*/
                    noInterrupts();
                    if (value & 1)
                    {
                        /*writes a value of 1 to the 0th bit*/
                        PORTF |= 128;
                    }
                    else
                    {
                        /*writes a value of 0 to the 0th bit*/
                        PORTF &= ~128;
                    }
                    interrupts();
                }
                if(!(pinConfigurations[2] & 2) || (DDRF & 64))
                {
                    /*analog pin 1 - PF6*/
                    /*we are good to configure the pin for writing, and write it's value*/
                    DDRF |= 64;
                    /*for safety turn off interrupts for the write operation*/
                    noInterrupts();
                    if (value & 2)
                    {
                        /*writes a value of 1 to the 1st bit*/
                        PORTF |= 64;
                    }
                    else
                    {
                        /*writes a value of 0 to the 1st bit*/
                        PORTF &= ~64;
                    }
                    interrupts();
                }
                if(!(pinConfigurations[2] & 4) || (DDRF & 32))
                {
                    /*analog pin 2 - PF5*/
                    /*we are good to configure the pin for writing, and write it's value*/
                    DDRF |= 32;
                    /*for safety turn off interrupts for the write operation*/
                    noInterrupts();
                    if (value & 4)
                    {
                        /*writes a value of 1 to the 2nd bit*/
                        PORTF |= 32;
                    }
                    else
                    {
                        /*writes a value of 0 to the 2nd bit*/
                        PORTF &= ~32;
                    }
                    interrupts();
                }
                if(!(pinConfigurations[2] & 8) || (DDRF & 16))
                {
                    /*analog pin 3 - PF4*/
                    /*we are good to configure the pin for writing, and write it's value*/
                    DDRF |= 16;
                    /*for safety turn off interrupts for the write operation*/
                    noInterrupts();
                    if (value & 8)
                    {
                        /*writes a value of 1 to the 3rd bit*/
                        PORTF |= 16;
                    }
                    else
                    {
                        /*writes a value of 0 to the 3rd bit*/
                        PORTF &= ~16;
                    }
                    interrupts();
                }
                if(!(pinConfigurations[2] & 16) || (DDRF & 2))
                {
                    /*analog pin 4 - PF1*/
                    /*we are good to configure the pin for writing, and write it's value*/
                    DDRF |= 2;
                    /*for safety turn off interrupts for the write operation*/
                    noInterrupts();
                    if (value & 16)
                    {
                        /*writes a value of 1 to the 2nd bit*/
                        PORTF |= 2;
                    }
                    else
                    {
                        /*writes a value of 0 to the 2nd bit*/
                        PORTF &= ~2;
                    }
                    interrupts();
                }
                if(!(pinConfigurations[2] & 32) || (DDRF & 1))
                {
                    /*analog pin 5 - PF0*/
                    /*we are good to configure the pin for writing, and write it's value*/
                    DDRF |= 1;
                    /*for safety turn off interrupts for the write operation*/
                    noInterrupts();
                    if (value & 32)
                    {
                        /*writes a value of 1 to the 1st bit*/
                        PORTF |= 1;
                    }
                    else
                    {
                        /*writes a value of 0 to the 1st bit*/
                        PORTF &= ~1;
                    }
                    interrupts();
                }
                /*there is not PC6 (actually it's the reset pin) and PC7*/
                /*done writing we can return*/
                return;
            default:
                /*wrong port, just return without doing anything*/
                return;
        }
    }
#endif
}


byte timingClassify(task * theTask)
{
	/*the different cases are:*/
	/*1. Infinite synced (time spent waiting in between calls)*/
	/*2. Infinite not synced (no time spent waiting in between calls)*/
	/*3. Time driven synced (will run in an infinte loop until a timer expires, pauses in between calls)*/
	/*4. Count driven synced (will run a set number of times with waiting in between calls)*/
	/*5. Time driven not synced (will run in an infinte loop until a timer expires, no pausing)*/
	/*6. Count driven not synced (will run a set number of times with no waiting in between calls)*/
	unsigned long syncTime = getSyncTime(theTask);
	/*Cases 1 and 2 are distinguished by the fact that both iterationCount and runTimeLength are 0*/
	if(getRunTimeLength(theTask) == 0 && getIterationCount(theTask) == 0)
	{
		/*Case 2 arises from syncTime being 0, while case 1 arises from sync time not being zero*/
		return syncTime ? 1 : 2;
	}
	/*if runTimeLength isn't zero, we have either case 3 or 5*/
	else if (getRunTimeLength(theTask))
	{
		/*case 3 arises from there being a non zero sync time, while case 5 requires no sync time*/
		return syncTime ? 3 : 5;
	}
	else
	{
		/*must be iteration count, or cases 4 or 6*/
		/*case 4 arises from a non zero sync time, while case 6 is a zero sync time*/
		return syncTime ? 4 : 6;
	}
}



/*GENERATED FUNCTION THAT CALLS A USER'S FUNCTION FOLLOWS*/

`callFunctionWithArguments`

/*END GENERATED FUNCTION THAT CALLS A USER'S FUNCTION*/

char * getArgumentsData(arguments * args)
{
	/*this will grab all the argument data information off the serial buffer*/
	/*first, we need to get the number of argument data packets to expect from the*/
	/*arguments struct that we are going to store this in*/
	int numArgumentPackets = getArgNum(args);
	/*now that we have that, for each packet, get the data and store it in the struct*/
	/*we also need a char array to store the arguments signature in and return*/
	char * argSignature = 0;
	argSignature = (char *)calloc((numArgumentPackets+1),sizeof(char));
	/*the plus one is for the null terminator, so we can set that now*/
	argSignature[numArgumentPackets+1]= '\0';
	
	long * fullLongs = 0;
	float * fullFloats = 0;
	
	/*these are for the array args*/
	char2DArray  * strings = 0;
	long2DArray * longs = 0;
	float2DArray * floats = 0;
	
	/*save these as local variables so we don't have to call the function repeatedly*/
	byte longArgNum = getLongArgNum(args);
	byte floatArgNum = getFloatArgNum(args);
	byte stringArgNum = getStringArgNum(args);
	byte longArrayArgNum = getLongArrayArgNum(args);
	byte floatArrayArgNum = getFloatArrayArgNum(args);
	/*for all of the long and float arguments, only create the array if there are any of those arguments*/
	if (longArgNum)
	{
		fullLongs = (long *)calloc(longArgNum,sizeof(long));
	}
	if (floatArgNum)
	{
		fullFloats = (float *)calloc(floatArgNum,sizeof(float));
	}
	/*for each 2d array, if there are any arguments to store there, allocate memory for it,*/
	/*and set the size of each*/
	if (stringArgNum)
	{
		strings = (char2DArray *)calloc(1,sizeof(char2DArray));
		setNumArraysChar(strings,stringArgNum);
	}
	if (longArrayArgNum)
	{
		longs = (long2DArray *)calloc(1,sizeof(long2DArray));
		setNumArraysLong(longs,longArrayArgNum);
	}
	if (floatArrayArgNum)
	{
		floats = (float2DArray *)calloc(1,sizeof(float2DArray));
		setNumArraysFloat(floats,floatArrayArgNum);
	}	
	/*the following variables are reused when appropriate in the for loop*/
	/*for keeping track of the total number of packets (this can be up to 16 longs + 16 floats + 4 long arrays + 4 float arrays + 32 strings)*/
	byte packetIndex;
	/*for keeping track of all of the individual long arguments (this can't be larger than 16)*/
	byte longIndex = 0;
	/*for getting all of the long numbers in a long array argument (this can't be larger than 255)*/
	byte longArrayIndex = 0;
	/*for keeping track of number of long arrays (can't be larger than 4)*/
	byte numLongArrays = 0;
	/*for getting all the float numbers in a float array argument (can't be larger than 255)*/
	byte floatArrayIndex = 0;
	/*for keeping track of all of the individual float arguments (this can't be larger than 16)*/
	byte floatIndex = 0;
	/*for keeping track of the number of float arrays (this can't be larger than 4)*/
	byte numFloatArrays = 0;
	/*for getting all the individual chars for the string arguments (this can't be larger than 255)*/
	byte charIndex = 0;
	/*for keeping track of the number of the string arguments (this can't be larger than 32)*/
	byte stringIndex = 0;

	/*for temporarily storing the bytes associated with a float or a long*/
	byte data3;
	unsigned int data2;
	unsigned long data1;
	unsigned long data0;
	/*for storing the array of numbers for an array arguments*/
	long * longNumbers = 0;
	/*for storing the array of numbers for an array arguments*/
	float * floatNumbers = 0;
	/*for storing the string*/
	char * string = 0;
	
	byte stringLength;
	byte floatArrayLength;
	byte longArrayLength;

	/*this union will be used to convert the raw bytes over the serial line to the floating point*/
	union Converter convert;

	for (packetIndex = 0; packetIndex < numArgumentPackets; packetIndex ++)
	{
		/*the first thing, is we need to make sure that the sysex packet is the first byte,*/
		/*if it's not, well then undefined behavior will occur, so we just won't do anything*/
		/*as always, we are expecting a byte, so wait for it to come*/
		while(SerialLink.available()<=0);
		if (SerialLink.read() == SYSEX_START)
		{
			/*now that byte is off the buffer, so we can switch case on the next one, which*/
			/*will tell us what kind of argument we are dealing with*/
			while(SerialLink.available()<=0);
			switch(SerialLink.read())
			{
				case LONG_NUM:
					/*this packet is a single long number*/
					/*we can guarantee that there are only 4 more data bytes, and one END_SYSEX byte*/
					while(SerialLink.available()<=0);
					data0 = SerialLink.read();
					while(SerialLink.available()<=0);
					data1 = SerialLink.read();
					while(SerialLink.available()<=0);
					data2 = SerialLink.read();
					while(SerialLink.available()<=0);
					data3 = SerialLink.read();
					/*the last byte is SYSEX_END, so just discard that*/
					while(SerialLink.available()<=0);
					SerialLink.read();
					convert.rawBytes = (data0 << 24) | (data1 << 16) | (data2 << 8) | data3;
					/*now use the union to reinterpret the data as a float*/
					fullLongs[longIndex] = convert.theLong;
					/*longIndex is the number of long args, so increment that here*/
					longIndex++;
					/*we can also set the arg signature string's element here*/
					argSignature[packetIndex] = '0';
					break;
				case FLOAT_NUM:
					/*this packet is a single floating point number*/
					/*we can guarantee that there are only 4 more data bytes, and one END_SYSEX byte*/
					while(SerialLink.available()<=0);
					data0 = SerialLink.read();
					while(SerialLink.available()<=0);
					data1 = SerialLink.read();
					while(SerialLink.available()<=0);
					data2 = SerialLink.read();
					while(SerialLink.available()<=0);
					data3 = SerialLink.read();
					/*the last byte is SYSEX_END, so just discard that*/
					while(SerialLink.available()<=0);
					SerialLink.read();
					/*the order here is dependent on the endianness of the arduino we are running on, but the*/
					/*client can simply send the bytes in the reverse order for a different endianness, so this won't need*/
					/*to change for different arduinos*/
					convert.rawBytes = (data0 << 24) | (data1 << 16) | (data2 << 8) | data3;
					/*now use the union to reinterpret the data as a float*/
					fullFloats[floatIndex] = convert.theFloat;
					/*floatIndex is the number of float args, so increment that here*/
					floatIndex++;
					/*we also need to set the arg signature string's element*/
					argSignature[packetIndex] = '1';
					break;
				case STRING:
					/*this packet is a string, or a character array*/
					/*the next byte will tell us how many bytes are in this array*/
					while(SerialLink.available()<=0);
					stringLength = SerialLink.read();
					/*now we can allocate memory for the string*/
					string = (char *)calloc((stringLength+1),sizeof(char));
					/*there is an implied null terminator, it isn't sent over the serial connection,*/
					/*but because it is known this is a string, we add it here*/
					string[stringLength] = '\0';
					/*we use a for loop to grab the all the letters in the string*/
					for (charIndex = 0; charIndex < stringLength; charIndex++)
					{
						while(SerialLink.available()<=0);
						string[charIndex] = SerialLink.read();
					}
					/*we also have to read off the SYSEX_END byte*/
					while(SerialLink.available()<=0);
					SerialLink.read();
					/*now the string is entirely saved inside the variable string, so we can put that*/
					/*inside the char2DArray strings*/
					setGivenArrayChar(strings,stringIndex,stringLength+1,string);
					/*now we can free the memory for the string*/
					free(string);
					/*stringIndex is the number of strings we have saved, so we need to increment it*/
					stringIndex++;
					/*we also need to set the arg signature string's element*/
					argSignature[packetIndex] = '2';
					break;
				case FLOAT_ARRAY:
					/*this packet is an array of floating point numbers*/
					/*the next byte will tell us how many floats are in this array*/
					while(SerialLink.available()<=0);
					floatArrayLength = SerialLink.read();
					/*now allocate the array*/
					floatNumbers = (float *)calloc(floatArrayLength,sizeof(float));
					/*we use a for loop to grab all the floats, the next four bytes are the float data, and they are combined*/
					/*into a long, then converted using the union convert to reinterpret the data*/
					for(floatArrayIndex = 0; floatArrayIndex < floatArrayLength; floatArrayIndex++)
					{
						while(SerialLink.available()<=0);
						data0=SerialLink.read();
						while(SerialLink.available()<=0);
						data1=SerialLink.read();
						while(SerialLink.available()<=0);
						data2=SerialLink.read();
						while(SerialLink.available()<=0);
						data3=SerialLink.read();
						convert.rawBytes = (data0 << 24) | (data1 << 16) | (data2 << 8) | data3;
						/*now use the union to reinterpret the data as a float*/
						floatNumbers[floatArrayIndex] = convert.theFloat;
					}
					/*we also have to read off the SYSEX_END byte*/
					while(SerialLink.available()<=0);
					SerialLink.read();
					/*now all the floats in this array are in floatNumbers, so we can put that*/
					/*inside the float2DArray*/
					setGivenArrayFloat(floats,numFloatArrays,floatArrayLength,floatNumbers);
					/*finally deallocate the memory for floatNumbers so it can be allocated with new memory when it is needed next*/
					free(floatNumbers);
					/*also increment the number of float arrays*/
					numFloatArrays++;
					/*we also need to set the arg signature string's element*/
					argSignature[packetIndex] = '3';
					break;
				case LONG_ARRAY:
					/*this packet is an array of long numbers*/
					/*the next byte will tell us how many longs are in this array*/
					while(SerialLink.available()<=0);
					longArrayLength = SerialLink.read();
					/*now we can allocate memory for the longNumbers array*/
					longNumbers = (long *)calloc(longArrayLength,sizeof(long));
					/*we use a for loop to grab all the longs, the first four bytes are then combined to make the full long*/
					for(longArrayIndex = 0; longArrayIndex < longArrayLength; longArrayIndex++)
					{
						while(SerialLink.available()<=0);
						data0=SerialLink.read();
						while(SerialLink.available()<=0);
						data1=SerialLink.read();
						while(SerialLink.available()<=0);
						data2=SerialLink.read();
						while(SerialLink.available()<=0);
						data3=SerialLink.read();
						convert.rawBytes = (data0 << 24) | (data1 << 16) | (data2 << 8) | data3;
						longNumbers[longArrayIndex] = convert.theLong;
					}
					/*we also have to read off the SYSEX_END byte*/
					while(SerialLink.available()<=0);
					SerialLink.read();
					/*now all the floats in this array are in floatNumbers, so we can put that*/
					/*inside the float2DArray*/
					setGivenArrayLong(longs,numLongArrays,longArrayLength,longNumbers);
					/*finally we need to free the memory so it can be allocated with new memory when it is needed next*/
					free(longNumbers);
					/*also increment the number of float arrays*/
					numLongArrays++;
					/*we also need to set the arg signature string's element*/
					argSignature[packetIndex] = '4';
					break;
				default:
					/*there isn't a default here, so again this will be undefined behavior...*/
					break;
			}
		}
	}
	/*now all the long numbers are stored in order inside fullLongs, all floats in fullFloats, etc.*/
	/*we can add these into the arguments struct now*/
	/*for each type, longs, 2d longs, etc. only perform the add operation if it exists. */
	/*if it doesn't exist, then the add operation will just copy a bunch of zeroes over and allocate unneeded memory*/
	if (longArgNum)
	{
		/*longIndex still has the number of long args*/
		setLongArgArray(args,fullLongs,longIndex);
		free(fullLongs);
	}
	if (floatArgNum)
	{
		/*floatIndex still has the number of float args*/
		setFloatArgArray(args,fullFloats,floatIndex);
		free(fullFloats);
	}
	if (stringArgNum)
	{
		setStringArgArray(args,strings);
		safeDeleteChar(strings);
	}
	if (longArrayArgNum)
	{
		setLongArrayArgArray(args,longs);
		safeDeleteLong(longs);
	}
	if (floatArrayArgNum)
	{
		setFloatArrayArgArray(args,floats);
		safeDeleteFloat(floats);
	}
	/*lastly return the arg signature that was built*/
	return argSignature;
}


/*GENERATED FUNCTION THAT VALIDATES THE FUNCTION ID OF A FUNCTION FOLLOWS*/

`functionID`

/*END GENERATED FUNCTION THAT VALIDATES THE FUNCTION ID OF A FUNCTION*/


task * getSerialFirmataTask(void)
{
	/*this function will read in from the SerialLink buffer enough bytes until it gets one full task.*/
	/*it will not disturb further bytes in the SerialLink buffer*/
	/*create the pointer for the return task, but don't allocate any memory for it until we know we need it,*/
	/*as we could have an errant byte in the serial buffer that doesn't correspond to anything meaningful.*/
	task * returnTask;
	/*arrayIndex is for iterating through any arrays from the SerialLink buffer*/
	int arrayIndex;
	/*first get what kind of task we are dealing with, which is represented by the first byte of any message*/
	byte fullTaskType = SerialLink.read();
	
	
	byte taskHighNibble = fullTaskType & 240;
	byte taskLowNibble = fullTaskType & 15;
	byte firstDataByte;
	int secondDataByte;
	byte pinValueBitMask;
	byte highAnalogVal;
	byte lowAnalogVal;
	int pinValue;
	byte hiddenPWMPinDisableBits;
	byte pin;
	byte configuration;
	byte functionID;
	byte longFloatArgNums;
	byte stringArrayArgNums;
	byte timingInfo;
	arguments * functionArgs = 0;
	byte longArgNum;
	byte floatArgNum;
	byte stringArgNum;
	byte longArrayArgNum;
	byte floatArrayArgNum;
	unsigned long data0;
	unsigned long data1;
	unsigned int data2;
	byte data3;
	char * argumentOrder;
	byte stringLength;
	int numElements;
	unsigned long timingLongNum;

	/*switch of the type of message, these are all #define'd above*/
	switch(taskHighNibble)
	{
		case DIGITAL_READ_TASK:
			/*the first byte of the digital message's high nibble is the digital read identifier, and */
			/*the low nibble is the port that is to be read. For example, the byte 0xC2 means to read*/
			/*from the third port, while 0xC0 means to read from the first port. The second byte is */
			/*the enable/disable bit, but this isn't used here, so we save it for possible future */
			/*functionality, but for now it will just get popped off the buffer and ignored*/
			/*NEW FEATURE: the three most significant bits in the low nibble of the second byte have "hidden" bits in them */
			/*corresponding to whether or not the PWM pins should be disabled in the read operation*/
			while(SerialLink.available()==0);
			firstDataByte = SerialLink.read();
			hiddenPWMPinDisableBits = (firstDataByte & ~1)>>1;
			/*check to make sure that the port number requested is valid for this board*/
			if(taskLowNibble<3)
			{
				returnTask = (task *)calloc(1,sizeof(task));
				/*still need to check to make sure that the allocation worked*/
				if (returnTask)
				{
					/*allocation succeeded, so we are good to start building the task*/
					/*the type of a digital read task is 2 (arbitrarily chosen)*/
					setType(returnTask,DIGITAL_READ);
					/*the port number is the low nibble of the first byte*/
					setPortNumber(returnTask,taskLowNibble);
					/*hide the hidden bits in the pin member of the struct*/
					setPinNumber(returnTask,hiddenPWMPinDisableBits);
					return returnTask;
				}
				else
				{
					/*allocation failed, so send an "OUT OF MEMORY" message over serial to*/
					/*let the host know that there is not enough memory and return a null pointer*/
					return 0;
				}
			}
			else
			{
				/*the user requested an invalid port, so return a null pointer*/
				return 0;
			}
		case DIGITAL_WRITE_TASK:
			/*the first byte of the digital message's high nibble is the digital write identifier, and the low nibble is*/
			/*the port to be written to. The pin's values are determined from the next two bytes, the first of which */
			/*contains the least significant 7 bits, while the last bit is contained in the first bit in the third byte*/
			/*the boolean values of the pins in that given port.*/
			/*because we are expecting bytes, we should wait until they get here before reading off the buffer*/
			while(SerialLink.available()<=0);
			firstDataByte = SerialLink.read();
			while(SerialLink.available()<=0);
			secondDataByte = SerialLink.read();
			pinValueBitMask = firstDataByte | ((secondDataByte & 1)<<7);
			hiddenPWMPinDisableBits = (secondDataByte & ~1)>>1;
			/*still need to check to make sure that the allocation worked*/
			if(taskLowNibble<3)
			{
				returnTask = (task *)calloc(1,sizeof(task));
				/*still need to check to make sure that the allocation worked*/
				if (returnTask)
				{
					/*allocation succeeded, so we are good to start building the task*/
					/*the type of a digital write task is 1 (arbitrarily chosen)*/
					setType(returnTask,DIGITAL_WRITE);
					/*the port number is the low nibble of the first byte*/
					setPortNumber(returnTask,taskLowNibble);
					/*the value of the pins in that port are given by the second byte*/
					setValue(returnTask,pinValueBitMask);
					/*the hidden bits from the value are for determining whether or not a write through to the PWM pins is intended, so we store that information in the pin byte*/
					setPinNumber(returnTask,hiddenPWMPinDisableBits);
					return returnTask;
				}
				else
				{
					/*allocation failed, so send an "OUT OF MEMORY" message over serial to*/
					/*let the host know that there is not enough memory and return a null pointer*/
					return 0;
				}
			}
			else
			{
				/*the user requested an invalid port, so return a null pointer*/
				return 0;
			}
		case ANALOG_WRITE_TASK:
			/*the first byte of the analog message's high nibble is the analog write identifier, and the low nibble is*/
			/*the pin to write to. The second and third bytes represent the value to write to the pin. The value is */
			/*sent as t 7-bit bytes, which means that the actual value is the last byte bit shifted to the left 7 bits,*/
			/*bitwise or'd with the second byte. This does enforce a maximum resolution of 14 bits.*/
			while(SerialLink.available()<=0);
			lowAnalogVal = SerialLink.read();
			while(SerialLink.available()<=0);
			highAnalogVal = SerialLink.read();
			pinValue = (highAnalogVal << 7) | lowAnalogVal;
			/*now that we know for certain that the task is valid, we can allocate memory for the task*/
			/*calloc will essentially zero initialize the task for us*/
			returnTask = (task *)calloc(1,sizeof(task));
			/*still need to check to make sure that the allocation worked*/
			if (returnTask)
			{
				/*allocation succeeded, so we are good to start building the task*/
				/*the type of an analog write task is 4*/
				setType(returnTask,ANALOG_WRITE);
				/*the pin number is the low nibble of the first byte*/
				setPinNumber(returnTask,taskLowNibble);
				/*the value to write is given by pinValue*/
				setValue(returnTask,pinValue);
				return(returnTask);
			}
			else
			{
				/*allocation failed, so send an "OUT OF MEMORY" message over serial to let the host know what*/
				/*happened and return a null pointer*/
				return 0;
			}
			/*no break needed here, both cases will return out of the function*/
		case ANALOG_READ_TASK:
			/*the first byte of the analog message's high nibble is the analog read identifier, and the low nibble is*/
			/*the pin to read from. The second byte is the enable/disable reporting value, but this is not used in this*/
			/*implementation, but maybe included in a future release. For now though, the second byte is basically popped*/
			/*off the serial buffer and promptly ignored.*/
			while(SerialLink.available()<=0);
			SerialLink.read();
			/*now that we know for certain that the task is valid, we can allocate memory for the task*/
			/*calloc will essentially zero initialize the task for us*/
			returnTask = (task *)calloc(1,sizeof(task));
			/*still need to check to make sure that the allocation worked*/
			if (returnTask)
			{
				/*allocation succeeded, so we are good to start building the task*/
				setType(returnTask,ANALOG_READ);
				/*the pin of the task is low nibble of the first byte*/
				setPinNumber(returnTask,taskLowNibble);
				return(returnTask);
			}
			else
			{
				/*allocation failed, so send an "OUT OF MEMORY" message over serial to let the host know what*/
				/*happened and return a null pointer*/
				return 0;
			}
		case COMMAND_TASK:
			/*there are a few different types of commands, so here we switch on the full byte to determine which*/
			/*kind of task*/
			
			switch(fullTaskType)
			{
				case CONFIG_TASK:
					/*config task will be able to set a pin to be input, output, etc.*/
					/*the breakdown of the bytes is that the second byte is the pin number (0-127),*/
					/*while the third byte is the state, which can be one of 9 different configurations*/
					/*at the moment, only input, output, analogInput, PWM (analogOutput) are supported,*/
					/*although the protocol and library allow for more including I2C,OneWire, Servo, etc.*/
					/*grab next byte which is the pin to configure*/
					while(SerialLink.available()<=0);
					pin = SerialLink.read();
					/*grab next byte which is the state to put the pin in*/
					while(SerialLink.available()<=0);
					configuration = SerialLink.read();
					/*allocate memory for the task here*/
					returnTask = (task *)calloc(1,sizeof(task));
					if(returnTask)
					{
						/*now need to add the pinNumber into the task*/
						setPinNumber(returnTask,pin);
						/*add the configuration of that pin in as the value*/
						setValue(returnTask,configuration);
						/*the type of a config task is 0*/
						setType(returnTask,0);
						return returnTask;
					}
					else
					{
						/*out of memory*/
						return 0;
					}
				case SYSEX_START:
					/*a sysex start byte means we have a sysex message to process, which could be any number*/
					/*of actual things*/
					/*first switch on the next byte, as that will uniquely determine which task we are dealing*/
					/*with*/
					while(SerialLink.available()<=0);
					switch(SerialLink.read())
					{
						/*the supported sysex commands for this version are as follows:*/
						/*(the first 5 aren't valid at this point in the code, as we don't have a task yet)*/
						/*LONG_NUM - a single long number*/
						/*FLOAT_NUM - a single float number*/
						/*STRING - an array of chars*/
						/*LONG_ARRAY - an array of longs*/
						/*FLOAT_ARRAY - an array of floats*/
						/*FUNCTION_CALL_ADD - adds the specified function to the system*/
						/*FUNCTION_CALL_DELETE - deletes the specified function call from the system*/
						/*FLUSH_SERIAL - deletes all other data off the serial buffer. ***ONLY FOR EMERGENCIES****/
						case FUNCTION_CALL_ADD:
							/*this should be the most frequent case*/
							/*before we start allocating memory, we need to check and make sure that a valid*/
							/*function was requested*/
							while(SerialLink.available()<=0);
							functionID = SerialLink.read();
							if(validFunctionID(functionID))
							{
								/*the function ID is valid, we can now allocate memory for the task and start building it, but first*/
								/*we are going to grab the rest off the data for this packet off of the SerialLink port*/
								/*the next byte in the packet for a function call task is the number of long and float arguments,*/
								/*stored as a bitmask, with the high nibble for the number of longs, and the low nibble for the number*/
								/*of floats*/
								while(SerialLink.available()<=0);
								longFloatArgNums = SerialLink.read();
								/*the next byte is the number of string args, long array args, and float array args*/
								while(SerialLink.available()<=0);
								stringArrayArgNums = SerialLink.read();
								/*the next byte in the packet is the timing info, which will determine what the next sysex packets are */
								while(SerialLink.available()<=0);
								timingInfo = SerialLink.read();
								/*the last byte is an END_SYSEX packet, so we just read it off*/
								/*maybe check it to see if it is actually an END_SYSEX packet*/
								while(SerialLink.available()<=0);
								SerialLink.read();
								returnTask = (task *)calloc(1,sizeof(task));
								/*still need to check to make sure that the allocation worked*/
								if (returnTask)
								{
									/*allocation succeeded, so we should allocate the arguments*/
									functionArgs = (arguments *)calloc(1,sizeof(arguments));
									/*check to make sure that allocation worked*/
									if(functionArgs)
									{
										/*the type of a function call task is FUNCTION_CALL*/
										setType(returnTask,FUNCTION_CALL);
										setID(returnTask,functionID);
										/*we don't add all the arguments information to the task, instead we put it inside an arguments adt, so we need*/
										/*to allocate that here first, then once we're done filling that up, we can put it inside the task, and delete it*/
										/*to put the number of long arguments in, we have to bitshift the byte 4 to the right*/
										longArgNum = longFloatArgNums >>4;
										setLongArgNum(functionArgs,longArgNum);
										/*for the float args, all we need to do is just bitwise and the byte with 15 (0b00001111)*/
										floatArgNum = longFloatArgNums&15;
										setFloatArgNum(functionArgs,floatArgNum);
										/*the high nibble of the stringArrayArgNum byte is the number of string args, */
										/*while the low nibble is the number of long and float array args*/
										/*to get the high nibble, we bit shift the byte down 4*/
										stringArgNum = stringArrayArgNums >> 4;
										/*the float array args is the least significant two bits in the low nibble, while the long array args*/
										/*is the most significant two bits in the high nibble*/
										longArrayArgNum = (stringArrayArgNums >> 2) & 3;
										floatArrayArgNum = stringArrayArgNums & 3;
										/*store these inside the function args struct*/
										setStringArgNum(functionArgs,stringArgNum);
										setLongArrayArgNum(functionArgs,longArrayArgNum);
										setFloatArrayArgNum(functionArgs,floatArrayArgNum);
										/*now we have to check to see if there are more packets coming in with information about the function task,*/
										/*not necessarily the arguments*/
										/*timingInfo will have as it's fifth bit, whether or not the function is a one time immediate function, if it is*/
										/*then we can skip all the timing info, and we will just run it, but we need to set the iteration count to 1,*/
										/*as iteration count of 0, and runTimeLength of 0 represents infinite times, and that would be what it defaults to*/
										/*if there is not more information*/
										/*there is some timing info to grab off the serial buffer, so grab the relevant packets*/
										/*note that this is the correct order they would come in, if they all existed*/
										/*note also that all these packets are long data types, so we can make assumptions about */
										/*how much data we need off the serial buffer*/
										if (timingInfo & 1)
										{
											/*then the next packet is how far in the future the task should be run, so we have to grab that packet first*/
											/*the first byte will be a sysex, in a future release, maybe check to make sure, but for now, just read it*/
											/*off*/
											while(SerialLink.available()<=0);
											SerialLink.read();
											/*the next byte is a long number identifier, so again, we can just ignore it*/
											while(SerialLink.available()<=0);
											SerialLink.read();
											/*the next four bytes are the actual data bytes, so we need to save those*/
											while(SerialLink.available()<=0);
											data0 = SerialLink.read();
											while(SerialLink.available()<=0);
											data1 = SerialLink.read();
											while(SerialLink.available()<=0);
											data2 = SerialLink.read();
											while(SerialLink.available()<=0);
											data3 = SerialLink.read();
											/*the last byte is SYSEX_END, so just discard that*/
											while(SerialLink.available()<=0);
											SerialLink.read();
											timingLongNum = (data0 << 24) + (data1 << 16) + (data2 << 8) + data3;
											setInitialWaitTime(returnTask,timingLongNum);
										}
										if (timingInfo & 2)
										{
											/*then the next packet is how much time is to be spent waiting in between tasks*/
											/*the first byte will be a sysex, in a future release, maybe check to make sure, but for now, just read it*/
											/*off*/
											while(SerialLink.available()<=0);
											SerialLink.read();
											/*the next byte is a long number identifier, so again, we can just ignore it*/
											while(SerialLink.available()<=0);
											SerialLink.read();
											/*the next four bytes are the actual data bytes, so we need to save those*/
											while(SerialLink.available()<=0);
											data0 = SerialLink.read();
											while(SerialLink.available()<=0);
											data1 = SerialLink.read();
											while(SerialLink.available()<=0);
											data2 = SerialLink.read();
											while(SerialLink.available()<=0);
											data3 = SerialLink.read();
											/*the last byte is SYSEX_END, so just discard that*/
											while(SerialLink.available()<=0);
											SerialLink.read();
											timingLongNum = (data0 << 24) + (data1 << 16) + (data2 << 8) + data3;
											setSyncTime(returnTask,timingLongNum);
										}
										if (timingInfo & 4)
										{
											/*then the next packet is how much time should elapse in total before the task expires*/
											/*the first byte will be a sysex, in a future release, maybe check to make sure, but for now, just read it*/
											/*off*/
											while(SerialLink.available()<=0);
											SerialLink.read();
											/*the next byte is a long number identifier, so again, we can just ignore it*/
											while(SerialLink.available()<=0);
											SerialLink.read();
											/*the next four bytes are the actual data bytes, so we need to save those*/
											while(SerialLink.available()<=0);
											data0 = SerialLink.read();
											while(SerialLink.available()<=0);
											data1 = SerialLink.read();
											while(SerialLink.available()<=0);
											data2 = SerialLink.read();
											while(SerialLink.available()<=0);
											data3 = SerialLink.read();
											/*the last byte is SYSEX_END, so just discard that*/
											while(SerialLink.available()<=0);
											SerialLink.read();
											timingLongNum = (data0 << 24) + (data1 << 16) + (data2 << 8) + data3;
											setRunTimeLength(returnTask,timingLongNum);
										}
										if (timingInfo & 8)
										{
											/*then the next packet is how many iterations should the task be run for before expiring*/
											/*the first byte will be a sysex, in a future release, maybe check to make sure, but for now, just read it*/
											/*off*/
											while(SerialLink.available()<=0);
											SerialLink.read();
											/*the next byte is a long number identifier, so again, we can just ignore it*/
											while(SerialLink.available()<=0);
											SerialLink.read();
											/*the next four bytes are the actual data bytes, so we need to save those*/
											while(SerialLink.available()<=0);
											data0 = SerialLink.read();
											while(SerialLink.available()<=0);
											data1 = SerialLink.read();
											while(SerialLink.available()<=0);
											data2 = SerialLink.read();
											while(SerialLink.available()<=0);
											data3 = SerialLink.read();
											/*the last byte is SYSEX_END, so just discard that*/
											while(SerialLink.available()<=0);
											SerialLink.read();
											timingLongNum = (data0 << 24) + (data1 << 16) + (data2 << 8) + data3;
											setIterationCount(returnTask,timingLongNum);
										}
										else
										{
											/*set the iteration count to 1 because there is no timing info, so we have to only*/
											/*call the function once, but we can't leave it zero, as a iteration count of 0*/
											/*and a time to run of zero corresponds to infinite task*/
											setIterationCount(returnTask,1);
										}
										/*now that the timing info for the function is all collected, we have to start getting the arguments' actual */
										/*data off the serial port*/
										/*the order in which the packets arrive is the order in which they should be added*/
										/*we call this function which will add all of the arguments data off the serial port into the arguments struct,*/
										/*then also return a char array of the size of the total amount of arguments, where each char represents the type*/
										/*of each argument*/
										/*not dynamically allocating this will make it one less thing we have to deallocate when we're done*/
										/*the plus one on the end is for the null terminator, so we can use it like an actual string*/
										/*but only get this if there are arguments available*/
										if (getArgNum(functionArgs))
										{
											argumentOrder = getArgumentsData(functionArgs);
											/*we also add the argument order inside the task*/
											/*setArgSignature(returnTask,argumentOrder,LENGTH(argumentOrder));*/
											/*then free argumentOrder*/
											free(argumentOrder);
										}
										/*now that we have the all the arguments data inside functionArgs, we can add it to the task*/
										setArgs(returnTask,functionArgs);
										/*make sure to safely delete the arguments built up in this function*/
										safeDeleteArguments(functionArgs);
										/*now we're good to go with building this task, we can just return the task*/
										return returnTask;
									}
									else
									{
										/*allocation failed for arguments, but not for task, so free task and return 0*/
										safeDeleteTask(returnTask);
										return 0;
									}
								}
								else
								{
									/*allocation failed, so send an "OUT OF MEMORY" message over serial to*/
									/*let the host know that there is not enough memory and return a null pointer*/
									safeDeleteArguments(functionArgs);
									return 0;
								}
							}
							else
							{
								/*invalid function ID, return a null pointer*/
								return 0;
							}
						case FUNCTION_CALL_DELETE:
							/*this will delete a given function call from the system*/
							/*the function call delete packet will have as its third byte (first being SYSEX_START, */
							/*the second being FUNCITON_CALL_DELETE) the unique number of the function call to stop*/
							/*before we allocate the memory, we need to check to make sure that a valid function is being*/
							/*requested. The funtion name is given by the third byte in the packet*/
							while(SerialLink.available()<=0);
							functionID = SerialLink.read();
							if(validFunctionID(functionID))
							{
								
								/*the function ID is valid, so we can now allocate memory for the task*/
								/*calloc will initialize it to zero for us*/
								returnTask = (task *)calloc(1,sizeof(task));
								/*also need to make sure to read the rest of the bytes off the serial buffer, for function*/
								/*call deletes, there is just one more byte, a SYSEX_END byte*/
								while(SerialLink.available()<=0);
								SerialLink.read();
								/*still need to check to make sure that the allocation worked*/
								if (returnTask)
								{
									/*allocation succeeded, so we are good to start building the task*/
									/*the type of a delete function call task is 6*/
									setType(returnTask,FUNCTION_DELETE);
									setID(returnTask,functionID);
									/*now return the task*/
									return returnTask;
								}
								else
								{
									/*allocation failed, so send an "OUT OF MEMORY" message over serial to*/
									/*let the host know that there is not enough memory and return a null pointer*/
									return 0;
								}

							}
							else
							{
								/*invalid function, so return a null pointer*/
								return 0;
							}
							/*no break needed here, both cases will return out of the function*/
						case STRING:
							/*this would be the first task packet on the serial buffer, but it is ambiguous*/
							/*as to what the string represents or is to be used for, so here, just grab it off*/
							/*the serial buffer and discard it basically*/
							/*the third byte on here is the length of the string, which corresponds directly*/
							/*to the number of bytes sent after that byte, and then one more for the end sysex */
							/*message*/
							while(SerialLink.available()<=0);
							stringLength = SerialLink.read();
							
							for (arrayIndex = 0; arrayIndex <= stringLength; arrayIndex++)
							{
								while(SerialLink.available()<=0);
								SerialLink.read();
							}
							/*once that data is off the buffer, return a null pointer*/
							return 0;
						case LONG_NUM:
							/*same deal as string, so just grab the data off the serial buffer and discard it*/
							/*for a long packet, there are only 5 more bytes, 4 for the number, and 1 for END_SYSEX*/
							/*so just read those packets off and discard them*/
							while(SerialLink.available()<=0);
							SerialLink.read();
							while(SerialLink.available()<=0);
							SerialLink.read();
							while(SerialLink.available()<=0);
							SerialLink.read();
							while(SerialLink.available()<=0);
							SerialLink.read();
							while(SerialLink.available()<=0);
							SerialLink.read();
							/*once those are removed from the serial buffer, return a null pointer*/
							return 0;
						case FLOAT_NUM:
							/*same deal as string, so just grab the data off the serial buffer and discard it*/
							/*for a float packet, there are only 5 more bytes, 4 for the number, and 1 for*/
							/*END_SYSEX*/
							while(SerialLink.available()<=0);
							SerialLink.read();
							while(SerialLink.available()<=0);
							SerialLink.read();
							while(SerialLink.available()<=0);
							SerialLink.read();
							while(SerialLink.available()<=0);
							SerialLink.read();
							while(SerialLink.available()<=0);
							SerialLink.read();
							/*once those are off the serial buffer, just return a null pointer*/
							return 0;
						case LONG_ARRAY:
							/*same deal as string, so just grab the data off the serial buffer and discard it*/
							/*for a long array packet, the third byte is the length of the array, so multiply*/
							/*the length by 4 and add 1 for END_SYSEX, and that's how many bytes are left*/
							while(SerialLink.available()<=0);
							numElements = SerialLink.read();
							for (arrayIndex=0; arrayIndex<(1+4*numElements); arrayIndex++)
							{
								while(SerialLink.available()<=0);
								SerialLink.read();
							}
							/*once that data is off the serial buffer, we can just return a null pointer*/
							return 0;
						case FLOAT_ARRAY:
							/*same deal as string, so just grab the data off the serial buffer and discard it*/
							/*the float array packet is structured exactly the same as the long array packet*/
							while(SerialLink.available()<=0);
							numElements = SerialLink.read();
							for (arrayIndex=0; arrayIndex<(1+4*numElements); arrayIndex++)
							{
								while(SerialLink.available()<=0);
								SerialLink.read();
							}
							/*once that data is off the serial buffer, we can just return a null pointer*/
							return 0;
						case FLUSH_SERIAL:
							/*this is for an emergency recovery way for the arduino to just clear all data off*/
							/*the serial buffer in case weird stuff starts happening. This should not be used*/
							/*hardly at all*/
							while(SerialLink.available())
							{
								SerialLink.read();
							}
							/*after the serial buffer is emptied, just return a null pointer*/
							return 0;
						default:
							/*this means an unsupported sysex message was either sent, or that there was junk on*/
							/*the serial line, so don't grab any more data off the serial line, just*/
							/*return a null pointer*/
							return 0;
					}
					break;
				default:
					/*there aren't any other kinds of messages that have a high COMMAND_TASK nibble,*/
					/*so return a null pointer to represent this is bad*/
					return 0;
			}
		default:
			/*should be no legitimate tasks with any*/
			/*other kind of high nibble, so just return a null pointer from the function in this case*/
			/*because we didn't do anything in the loop, but we still made an arguments adt, we need to safely delete it*/
			return 0;
	}
}







void runTask(task * functionTask, byte * pinConfigurations)
{
	/*switch on which kind we are dealing with,*/
	/*one of the following:*/
	/*1. Infinite synced (time spent waiting in between calls)*/
	/*2. Infinite not synced (no time spent waiting in between calls)*/
	/*3. Time driven synced (will run in an infinte loop until a timer expires, pauses in between calls)*/
	/*4. Count driven synced (will run a set number of times with waiting in between calls)*/
	/*5. Time driven not synced (will run in an infinte loop until a timer expires, no pausing)*/
	/*6. Count driven not synced (will run a set number of times with no waiting in between calls)*/

	/*newTask is for "interrupt" handling between iterations of function calls*/
	task * newTask;
	byte type = timingClassify(functionTask);
	unsigned long callCount;
	unsigned long interval;
	unsigned long startTime;
	unsigned long previousTime = 0;
	unsigned long endTime;
	switch(type)
	{
		case 1:
			/*infinite synced case*/
			interval = getSyncTime(functionTask);
			/*call the function once, as it must be called at least once*/
			/*this will add some time overhead to calling the function, but it makes*/
			/*generating the code much easier, especially for handling the case of multiple functions with different arguments*/
			callFunctionWithArguments(functionTask);
			previousTime = millis();
			while(1)
			{
				/*infinite loop*/
				/*get current time*/
				/*check if it is time to call the function*/
				if(millis() - previousTime >= interval)
				{
					/*it's time to call it again*/
					callFunctionWithArguments(functionTask);
					/*make sure to reset the timer*/
					previousTime = millis();
				}
				/*next check SerialLink buffer for new tasks*/
				if(SerialLink.available())
				{
					/*there's new data on the SerialLink port*/
					newTask = getSerialFirmataTask();
					/*check to make sure it is a legit task*/
					if (newTask)
					{
						/*it is a legit task, switch on the type*/
						switch(getType(newTask))
						{
							case 0:
								/*config task*/
								/*first run the task*/
								configTaskRun(newTask,pinConfigurations);
								/*then safely delete the data*/
								safeDeleteTask(newTask);
								break;
							case DIGITAL_WRITE:
								/*digital write task*/
								/*the code for the digital write task is the same for the analog write task*/
							case ANALOG_WRITE:
								/*analog write task*/
								/*first run the task*/
								writeTaskRun(newTask,pinConfigurations);
								/*then safely delete the data*/
								safeDeleteTask(newTask);
								break;
							case DIGITAL_READ:
								/*digital read task*/
								/*the code for the digital read task is the same for the analog read task*/
							case ANALOG_READ:
								/*analog read task*/
								/*first run the task*/
								readTaskRun(newTask,pinConfigurations);
								/*then safely delete the data*/
								safeDeleteTask(newTask);
								break;
							case FUNCTION_CALL:
								/*function call task*/
								/*NOTE: CURRENTLY UNIMPLEMENTED, so the task just gets ignored, but we still free it here*/
								safeDeleteTask(newTask);
								break;
							case FUNCTION_DELETE:
								/*function delete/quit task*/
								/*make sure the delete task is for this task, if it's not run the subroutine for deleting the task from the appropriate */
								/*queue later*/
								if(getID(newTask) == getID(functionTask))
								{
									/*it's for this function, so we can quit out of this function, but first we need to safely delete all the data*/
									/*for the current task*/
									safeDeleteTask(newTask);
									/*we will let the calling function delete the pointer to the functionTask*/
									return;
								}
								break;
							default:
								/*there should be no other type of function, so just continue here*/
								break;
						}/*end switch*/
					}/*end null pointer check*/
					else
					{
						/*it's a null pointer, continue to the next iteration*/
						continue;
					}
				}/*end SerialLink.available()*/
			}/*end infinite while loop*/
			return;
		case 2:
			/*infinite not synced*/
			while(1)
			{
				callFunctionWithArguments(functionTask);
				/*next check SerialLink buffer for new tasks*/
				if(SerialLink.available())
				{
					/*there's new data on the SerialLink port*/
					newTask = getSerialFirmataTask();
					/*check to make sure it is a legit task*/
					if (newTask)
					{
						/*it is a legit task, switch on the type*/
						switch(getType(newTask))
						{
							case 0:
								/*config task*/
								/*first run the task*/
								configTaskRun(newTask,pinConfigurations);
								/*then safely delete the data*/
								safeDeleteTask(newTask);
								break;
							case DIGITAL_WRITE:
								/*digital write task*/
								/*the code for the digital write task is the same for the analog write task*/
							case ANALOG_WRITE:
								/*analog write task*/
								/*first run the task*/
								writeTaskRun(newTask,pinConfigurations);
								/*then safely delete the data*/
								safeDeleteTask(newTask);
								break;
							case DIGITAL_READ:
								/*digital read task*/
								/*the code for the digital read task is the same for the analog read task*/
							case ANALOG_READ:
								/*analog read task*/
								/*first run the task*/
								readTaskRun(newTask,pinConfigurations);
								/*then safely delete the data*/
								safeDeleteTask(newTask);
								break;
							case 5:
								/*function call task*/
								/*NOTE: CURRENTLY UNIMPLEMENTED, so the task just gets ignored, but we still free it here*/
								safeDeleteTask(newTask);
								break;
							case FUNCTION_DELETE:
								/*function delete/quit task*/
								/*make sure the delete task is for this task, if it's not run the subroutine for deleting the task from the appropriate */
								/*queue later*/
								if(getID(newTask) == getID(functionTask))
								{
									/*it's for this function, so we can quit out of this function, but first we need to safely delete all the data*/
									/*for the current task*/
									safeDeleteTask(newTask);
									/*we will let the calling function delete the pointer to the functionTask*/
									return;
								}
								break;
							default:
								/*there should be no other type of function, so just continue here*/
								break;
						}/*end switch*/
					}/*end null pointer check*/
					else
					{
						/*it's a null pointer, continue to the next iteration*/
						continue;
					}
				}/*end SerialLink.available()*/
			}/*end infinite while loop*/
			return;
		case 3:
			/*time driven synced*/
			
			interval = getSyncTime(functionTask);
			endTime = getRunTimeLength(functionTask);
			/*set the start time before entering the loop*/
			startTime = millis();
			while(millis()- startTime < endTime)
			{
				if(millis() - previousTime >= interval)
				{
					/*it's time to call it again*/
					callFunctionWithArguments(functionTask);
					/*make sure to reset the timer*/
					previousTime = millis();
				}
				/*next check SerialLink buffer for new tasks*/
				if(SerialLink.available())
				{
					/*there's new data on the SerialLink port*/
					newTask = getSerialFirmataTask();
					/*check to make sure it is a legit task*/
					if (newTask)
					{
						/*it is a legit task, switch on the type*/
						switch(getType(newTask))
						{
							case 0:
								/*config task*/
								/*first run the task*/
								configTaskRun(newTask,pinConfigurations);
								/*then safely delete the data*/
								safeDeleteTask(newTask);
								break;
							case DIGITAL_WRITE:
								/*digital write task*/
								/*the code for the digital write task is the same for the analog write task*/
							case ANALOG_WRITE:
								/*analog write task*/
								/*first run the task*/
								writeTaskRun(newTask,pinConfigurations);
								/*then safely delete the data*/
								safeDeleteTask(newTask);
								break;
							case DIGITAL_READ:
								/*digital read task*/
								/*the code for the digital read task is the same for the analog read task*/
							case ANALOG_READ:
								/*analog read task*/
								/*first run the task*/
								readTaskRun(newTask,pinConfigurations);
								/*then safely delete the data*/
								safeDeleteTask(newTask);
								break;
							case 5:
								/*function call task*/
								/*NOTE: CURRENTLY UNIMPLEMENTED, so the task just gets ignored, but we still free it here*/
								safeDeleteTask(newTask);
								break;
							case FUNCTION_DELETE:
								/*function delete/quit task*/
								/*make sure the delete task is for this task, if it's not run the subroutine for deleting the task from the appropriate */
								/*queue later*/
								if(getID(newTask) == getID(functionTask))
								{
									/*it's for this function, so we can quit out of this function, but first we need to safely delete all the data*/
									/*for the current task*/
									safeDeleteTask(newTask);
									/*we will let the calling function delete the pointer to the functionTask*/
									return;
								}
								break;
							default:
								/*there should be no other type of function, so just continue here*/
								break;
						}/*end switch*/
					}/*end null pointer check*/
					else
					{
						/*it's a null pointer, continue to the next iteration*/
						continue;
					}
				}/*end SerialLink.available()*/
			}/*end time while loop*/
			return;
		case 4:
			/*count driven synced*/
			previousTime = 0;
			interval = getSyncTime(functionTask);
			/*the compiler can optimize this for loop much faster if we loop down with callCount-- as the condition, it will stop when it */
			/*hits zero*/
			for (callCount = getIterationCount(functionTask); callCount--;)
			{
				/*we put this inside an infinite while loop, because we basically want to */
				/*loop through the checking code for SerialLink.available() until the time has*/
				/*elapsed, and once it has, then we can get to the next iteration*/
				while(1)
				{
					if(millis() - previousTime >= interval)
					{
						/*it's time to call it again*/
						callFunctionWithArguments(functionTask);
						/*make sure to reset the timer*/
						previousTime = millis();
						/*now we can break out of the infinite while loop*/
						break;
					}
					/*next check SerialLink buffer for new tasks*/
					/*also make sure that this isn't the last iteration, if it is then we shouldn't grab data off the serial buffer*/
					if(SerialLink.available()&&callCount>1)
					{
						/*there's new data on the SerialLink port*/
						newTask = getSerialFirmataTask();
						/*check to make sure it is a legit task*/
						if (newTask)
						{
							/*it is a legit task, switch on the type*/
							switch(getType(newTask))
							{
								case 0:
									/*config task*/
									/*first run the task*/
									configTaskRun(newTask,pinConfigurations);
									/*then safely delete the data*/
									safeDeleteTask(newTask);
									break;
								case DIGITAL_WRITE:
									/*digital write task*/
									/*the code for the digital write task is the same for the analog write task*/
								case ANALOG_WRITE:
									/*analog write task*/
									/*first run the task*/
									writeTaskRun(newTask,pinConfigurations);
									/*then safely delete the data*/
									safeDeleteTask(newTask);
									break;
								case DIGITAL_READ:
									/*digital read task*/
									/*the code for the digital read task is the same for the analog read task*/
								case ANALOG_READ:
									/*analog read task*/
									/*first run the task*/
									readTaskRun(newTask,pinConfigurations);
									/*then safely delete the data*/
									safeDeleteTask(newTask);
									break;
								case 5:
									/*function call task*/
									/*NOTE: CURRENTLY UNIMPLEMENTED, so the task just gets ignored, but we still free it here*/
									safeDeleteTask(newTask);
									break;
								case FUNCTION_DELETE:
									/*function delete/quit task*/
									/*make sure the delete task is for this task, if it's not run the subroutine for deleting the task from the appropriate */
									/*queue later*/
									if(getID(newTask) == getID(functionTask))
									{
										/*it's for this function, so we can quit out of this function, but first we need to safely delete all the data*/
										/*for the current task*/
										safeDeleteTask(newTask);
										/*we will let the calling function delete the pointer to the functionTask*/
										return;
									}
									break;
								default:
									/*there should be no other type of function, so just continue here*/
									break;
							}/*end switch*/
						}/*end null pointer check*/
						else
						{
							/*it's a null pointer, continue to the next iteration*/
							continue;
						}
					}/*end SerialLink.available()*/
				}/*end infinite while loop*/
			}
			return;
		case 5:
			/*time driven not synced*/
			endTime = getRunTimeLength(functionTask);
			/*call the function once before entering the while loop*/
			startTime = millis();
			while(millis()- startTime < endTime)
			{
				callFunctionWithArguments(functionTask);
				/*next check SerialLink buffer for new tasks*/
				if(SerialLink.available())
				{
					
					/*there's new data on the SerialLink port*/
					newTask = getSerialFirmataTask();
					/*check to make sure it is a legit task*/
					if (newTask)
					{
						/*it is a legit task, switch on the type*/
						switch(getType(newTask))
						{
							case 0:
								/*config task*/
								/*first run the task*/
								configTaskRun(newTask,pinConfigurations);
								/*then safely delete the data*/
								safeDeleteTask(newTask);
								break;
							case DIGITAL_WRITE:
								/*digital write task*/
								/*the code for the digital write task is the same for the analog write task*/
							case ANALOG_WRITE:
								/*analog write task*/
								/*first run the task*/
								writeTaskRun(newTask,pinConfigurations);
								/*then safely delete the data*/
								safeDeleteTask(newTask);
								break;
							case DIGITAL_READ:
								/*digital read task*/
								/*the code for the digital read task is the same for the analog read task*/
							case ANALOG_READ:
								/*analog read task*/
								/*first run the task*/
								readTaskRun(newTask,pinConfigurations);
								/*then safely delete the data*/
								safeDeleteTask(newTask);
								break;
							case 5:
								/*function call task*/
								/*NOTE: CURRENTLY UNIMPLEMENTED, so the task just gets ignored, but we still free it here*/
								safeDeleteTask(newTask);
								break;
							case FUNCTION_DELETE:
								/*function delete/quit task*/
								/*make sure the delete task is for this task, if it's not run the subroutine for deleting the task from the appropriate */
								/*queue later*/
								if(getID(newTask) == getID(functionTask))
								{
									/*it's for this function, so we can quit out of this function, but first we need to safely delete all the data*/
									/*for the current task*/
									safeDeleteTask(newTask);
									/*we will let the calling function delete the pointer to the functionTask*/
									return;
								}
								break;
							default:
								/*there should be no other type of function, so just continue here*/
								break;
						}/*end switch*/
					}/*end null pointer check*/
					else
					{
						/*it's a null pointer, continue to the next iteration*/
						continue;
					}
				}/*end SerialLink.available()*/
			}/*end time while loop*/
			/*done running this task, but we don't need to worry about deleting functionTask, that will be taken care of outside of this function*/
			return;
		case 6:
			/*count driven not synced*/
			/*the compiler can optimize this for loop much faster if we loop down with callCount-- as the condition, it will stop when it */
			/*hits zero*/
			for (callCount = getIterationCount(functionTask); callCount--;)
			{
				callFunctionWithArguments(functionTask);
				/*next check SerialLink buffer for new tasks*/
				/*only get the task off if we aren't on the last iteration*/
				if(SerialLink.available()&&callCount>1)
				{
					/*there's new data on the SerialLink port*/
					newTask = getSerialFirmataTask();
					/*check to make sure it is a legit task*/
					if (newTask)
					{
						/*it is a legit task, switch on the type*/
						switch(getType(newTask))
						{
							case 0:
								/*config task*/
								/*first run the task*/
								configTaskRun(newTask,pinConfigurations);
								/*then safely delete the data*/
								safeDeleteTask(newTask);
								break;
							case DIGITAL_WRITE:
								/*digital write task*/
								/*the code for the digital write task is the same for the analog write task*/
								/*so just continue down*/
							case ANALOG_WRITE:
								/*analog write task*/
								/*first run the task*/
								writeTaskRun(newTask,pinConfigurations);
								/*then safely delete the data*/
								safeDeleteTask(newTask);
								break;
							case DIGITAL_READ:
								/*digital read task*/
								/*the code for the digital read task is the same for the analog read task,*/
								/*so just continue down*/
							case ANALOG_READ:
								/*analog read task*/
								/*first run the task*/
								readTaskRun(newTask,pinConfigurations);
								/*then safely delete the data*/
								safeDeleteTask(newTask);
								break;
							case 5:
								/*function call task*/
								/*NOTE: CURRENTLY UNIMPLEMENTED, so the task just gets ignored, but we still free it here*/
								safeDeleteTask(newTask);
								break;
							case FUNCTION_DELETE:
								/*function delete/quit task*/
								/*make sure the delete task is for this task, if it's not run the subroutine for deleting the task from the appropriate */
								/*queue later*/
								if(getID(newTask) == getID(functionTask))
								{
									/*it's for this function, so we can quit out of this function, but first we need to safely delete all the data*/
									/*for the current task*/
									safeDeleteTask(newTask);
									/*we will let the calling function delete the pointer to the functionTask*/
									return;
								}
								break;
							default:
								/*there should be no other type of function, so just continue here*/
								break;
						}/*end switch*/
					}/*end null pointer check*/
					else
					{
						/*it's a null pointer, continue to the next iteration*/
						continue;
					}
				}/*end SerialLink.available()*/
			}
			return;
		default:
			/*there isn't another kind of task to run, so just return*/
			return;
	}
}



#endif	/* TASK_H */



/**
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
===========================================================================================================
*/




/* 
 * File:   queue.h
 * Author: Ian
 *
 * Created on January 19, 2015, 10:07 PM
 */

#ifndef QUEUE_H
#define	QUEUE_H








/** taskQueueEmpty - returns a boolean of whether or not the task is empty
 * 
 *  Note: if it is not empty, the size will be returned.
 * 
 * @param theQueue - queue to check the size of
 * @return the size of the queue if not empty, zero otherwise
 */
inline byte taskQueueEmpty(queue * theQueue)
{
    return theQueue->size;
}




/** createQueue allocates memory for a queue struct
 * 
 * @param theQueue the queue to configure
 * @return - the created queue
 */
queue * createQueue()
{
    queue * theQueue;
    /*allocate memory for the entire queue first*/
    theQueue = (queue *) calloc(1,sizeof(queue));
    /*check to make sure it worked before allocating memory for the first and*/
    /*last nodes*/
    if (theQueue)
    {
        /*if the allocation worked, then set the size, the first add operation*/
        /*will set the appropriate first node*/
        theQueue->size = 0;
    }
    return theQueue;
}


void insertNextNode(node * currentNode, node * nextNode)
{
    /*first we are going to make a copy of the newNode to be added, so that*/
    /*the user can free their version of the pointer safely once we add the*/
    /*pointer in*/
    node * newNode = (node *)calloc(1,sizeof(node));
    newNode->data = (task *)calloc(1,sizeof(task));
    /*copy the task data from the nextNode to the newNode*/
    taskCopy(newNode->data,nextNode->data);
    /*now set the previous node of the new node to be the next node of the*/
    /*current node*/
    newNode->next = currentNode->next;
    /*then set the next node of the new node to be the current node*/
    newNode->previous = currentNode;
    /*then set the previous node of the current node to be the new node*/
    currentNode->next = newNode;
    /*if the current node's previous node was null, we are done, however, if */
    /*it wasn't null, we still have to point the previous node at the new node*/
    if (newNode->next)
    {
        (newNode->next)->previous = newNode;
    }
}



/** safeDeleteNode will safely delete the task stored inside the node, but DOES
 *      NOT safely delete the next and previous nodes inside it, so it is
 *      assumed that the node has already been properly removed from the list
 * 
 *  Note: the pointer passed in is also not freed, that must be done outside
 *      this function
 * 
 * @param theNode - the node to safely delete
 */
void safeDeleteNode(node * theNode)
{
	if(theNode)
	{
		/*all we basically need to do is perform a safe delete of the task inside*/
		/*the data*/
		safeDeleteTask(theNode->data);
		/*make sure not to free the memory that is pointed to by the node's next*/
		/*and previous data, that data could still be in the queue!*/
	}
}


/** taskQueueNodeEnqueue - adds the node to the back of the queue
 * 
 *  Note: this function will allocate new memory for the chain's reference, so
 *      the user needs to free their node after inserting
 * 
 * @param theQueue
 * @param newNode
 */
void taskQueueNodeEnqueue(queue * theQueue, node * newTaskNode)
{
    /*first allocate memory for a new node to add to the back*/
    node * newNode = (node *) calloc(1,sizeof(node));
    /*make sure the allocation succeeded*/
    if(newNode)
    {
        /*allocation worked, so we are good to copy over the task from the */
        /*task the user provided to the new task inside the new node we just */
        /*made*/
        newNode->data = (task *)calloc(1,sizeof(task));
        if (newNode->data)
        {
            /*the allocation for the task worked*/
            taskCopy(newNode->data,newTaskNode->data);
			/*save this as a local variable*/
			byte taskQueueSize = taskQueueEmpty(theQueue);
            /*case 1: the queue is empty*/
            if (!taskQueueSize)
            {
                /*it is empty, so we need to set the first node to be the new node*/
                theQueue->first = newNode;
                /*sanity check on the first node*/
                (theQueue->first)->next = 0;
            }
            /*case 2: the queue has one node, at the front (note it can't have only*/
            /*one node at the back)*/
            else if (taskQueueSize==1)
            {
                /*the queue has one element, and that element will be the first node*/
                /*so, set the first node's previous node to be the new node*/
                (theQueue->first)->previous = newNode;
                /*then set the next node for the new node to be the first node*/
                newNode->next = theQueue->first;
                /*finally set the last node of the queue to be the new node*/
                theQueue->last = newNode;
            }
            /*case 3: more than 1 node*/
            else
            {
                /*normal case of more than one node*/
                /*point the next node to the new node to be the old last node*/
                newNode->next = theQueue->last;
                /*point the old last node to the new node*/
                (theQueue->last)->previous = newNode;
                /*finally update the last node in the queue*/
                theQueue->last = newNode;
                /*sanity check on the last node*/
                (theQueue->last)->previous = 0;
            }

            /*note that we don't free or safely delete the newNode, as it is */
            /*safely inside the queue and it will stay there until the user*/
            /*dequeues it from the queue, at which point we can dequeue it*/
            /*lastly, update the queue's length*/
            (theQueue->size)++;
        }
    }
}


/** taskQueueEnqueue - appends the task to the back of the queue
 * 
 * @param theQueue - the queue to configure
 * @param newTask - the task to enqueue
 */
void taskQueueEnqueue(queue * theQueue, task * newTask)
{
    /*first allocate memory for a new node to add to the back*/
    node * newNode = (node *) calloc(1,sizeof(node));
    /*make sure the allocation succeeded*/
    if(newNode)
    {
        /*allocation worked, so we are good to copy over the task from the */
        /*task the user provided to the new task inside the new node we just */
        /*made*/
        newNode->data = (task *)calloc(1,sizeof(task));
        if (newNode->data)
        {
            /*the allocation for the task worked*/
            taskCopy(newNode->data,newTask);
			/*save the size as a local variable to prevent multiple function calls*/
			byte taskQueueSize = taskQueueEmpty(theQueue);
            /*case 1: the queue is empty*/
            if (!taskQueueSize)
            {
                /*it is empty, so we need to set the first node to be the new node*/
                theQueue->first = newNode;
				/*sanity check*/
				(theQueue->first)->next = 0;
				(theQueue->first)->previous = 0;
				(theQueue->last)=0;
            }
            /*case 2: the queue has one node, at the front (note it can't have only*/
            /*one node at the back)*/
            else if (taskQueueSize==1)
            {
                /*the queue has one element, and that element will be the first node*/
                /*so, set the first node's previous node to be the new node*/
                (theQueue->first)->previous = newNode;
                /*then set the next node for the new node to be the first node*/
                newNode->next = theQueue->first;
                /*finally set the last node of the queue to be the new node*/
                theQueue->last = newNode;
            }
            /*case 3: more than 1 node*/
            else
            {
                /*normal case of more than one node*/
                /*point the next node to the new node to be the old last node*/
                newNode->next = theQueue->last;
                /*point the old last node to the new node*/
                (theQueue->last)->previous = newNode;
                /*finally update the last node in the queue*/
                theQueue->last = newNode;
                /*sanity check on the last node*/
                (theQueue->last)->previous = 0;
            }

            /*note that we don't free or safely delete the newNode, as it is */
            /*safely inside the queue and it will stay there until the user*/
            /*dequeues it from the queue, at which point we can dequeue it*/
            /*lastly, update the queue's length*/
            (theQueue->size)++;
        }
    }
}


/** taskQueueDequeue - dequeues a task off the task queue, and returns it
 * 
 * @param theQueue - the queue to get the task off
 * @return a pointer to a newly allocated task
 */
task * taskQueueDequeue(queue * theQueue)
{
    /*first we will need to get a pointer to the task at the front of the queue*/
    task * returnTask;
    returnTask = (task *)calloc(1,sizeof(task));
    /*check to make sure the allocation worked*/
    if (returnTask)
    {
		/*it worked, so copy the task over*/
		taskCopy(returnTask,(theQueue->first)->data);
        /*now we can delete the node inside the queue safely*/
        safeDeleteNode(theQueue->first);
        /*null the next pointer for the second node in the chain so that it */
        /*will be the new front of the linked list*/
        ((theQueue->first)->previous)->next = 0;
        /*save a temp reference to the first node's previous node, so we can free*/
        /*the first node*/
        node * tempLast = (theQueue->first)->previous;
        /*then we can null the reference to the previous node in the first node*/
        (theQueue->first)->previous = 0;
        /*then free the pointer to the first node*/
        free(theQueue->first);
        /*then lastly set the first node to the temp pointer we just saved of */
        /*the old second node*/
        theQueue->first = tempLast;
        /*then decrement the size of the queue*/
        (theQueue->size)--;
        /*finally we can return to the user the returnTask*/
        return returnTask;
    }
	else
	{
		/*memory allocation failed*/
		return 0;
	}
}



/** frontNodeEnqueue - adds the node to the front of the queue
 * 
 *  Note: a new node is allocated for the queue, so the one passed in must be 
 *      freed, but it is not freed inside here, that must be done elsewhere
 * 
 * @param theQueue - the queue to configure
 * @param newFrontNode - the node to insert in front
 */
void frontNodeEnqueue(queue * theQueue, node * newFrontNode)
{
    /*all we need to do here is set the queue's first node to the new one,*/
    /*and set the next node for the new one to be the old first one*/
    /*first though, we should copy over the data from inside the newFrontNode*/
    /*first allocate memory for a new node to add to the back*/
    node * newNode = (node *) calloc(1,sizeof(node));
    /*make sure the allocation succeeded*/
    if(newNode)
    {
        /*allocation worked, so we are good to copy over the task from the */
        /*task the user provided to the new task inside the new node we just */
        /*made*/
        newNode->data = (task *)calloc(1,sizeof(task));
        if (newNode->data)
        {
            /*the allocation for the task worked*/
            taskCopy(newNode->data,newFrontNode->data);
			/*save the size as a local variable to prevent multiple function calls*/
			byte taskQueueSize = taskQueueEmpty(theQueue);
            /*case 1: the queue is empty*/
            if (!taskQueueSize)
            {
                /*it is empty, so we need to set the first node to be the new node*/
                theQueue->first = newNode;
                /*sanity check*/
                (theQueue->first)->next = 0;
            }
            /*case 2: the queue has one node, at the front (note it can't have only*/
            /*one node at the back)*/
            else if (taskQueueSize==1)
            {
                /*there is only one node, at the front, so first point that node*/
                /*at the new node*/
                (theQueue->first)->next = newNode;
                /*then set the previous node for the new node to be the first node*/
                newNode->previous = theQueue->first;
                /*set the last node of the queue to be the old front node*/
                theQueue->last = theQueue->first;
                /*finally set the first node of the queue to be the new node*/
                theQueue->first = newNode;
                /*sanity check*/
                (theQueue->first)->next = 0;
                (theQueue->last)->previous = 0;
            }
            /*case 3: more than 1 node*/
            else
            {
                /*normal case of more than one node*/
                /*point the previous node to the new node to be the old first node*/
                newNode->previous = theQueue->first;
                /*point the old first node to the new node*/
                (theQueue->first)->next = newNode;
                /*finally update the first node in the queue to be the new one*/
                theQueue->first = newNode;
                /*sanity check on the last node*/
                (theQueue->first)->next = 0;
            }

            /*note that we don't free or safely delete the newNode, as it is */
            /*safely inside the queue and it will stay there until the user*/
            /*dequeues it from the queue, at which point we can dequeue it*/
            /*lastly, update the queue's length*/
            (theQueue->size)++;
        }
    }
}




/** taskScheduledQueueAdd will insert the given task in order based on how
 *      long before the task is set to run
 * 
 *  Note: uses insertion sort, as the queue is implemented as a doubly linked 
 *      list
 * 
 *  Note: also assumes that the queue passed in is in sorted order, undefined
 *      behavior will occur if the queue is not sorted
 * 
 * @param scheduledQueue - the queue to configure
 * @param newTask - the task to add in
 */
void taskScheduledQueueAdd(queue * scheduledQueue, task * newTask)
{
    /*before we add anything, we need to see if the queue is empty, if */
    /*it is we don't need to do any special sorting algorithm*/
    if (!taskQueueEmpty(scheduledQueue))
    {
        /*the queue is empty, so just add in the task normally*/
        taskQueueEnqueue(scheduledQueue,newTask);
        return;
    }
    /*first we need to create a new node with the new task as it's data*/
    node * newNode = (node *) calloc(1,sizeof(node));
    /*check to make sure the allocation worked*/
    if(newNode)
    {
        /*allocation worked, so we can continue with configuring the task*/
        /*copy the contents of the newTask pointer to the new node's data*/
        /*pointer*/
        /*now allocate memory for the task itself*/
        newNode->data = (task *)calloc(1,sizeof(task));
        if (newNode->data)
        {
            /*allocation for the task worked*/
            taskCopy(newNode->data,newTask);
            /*now, we can add the task into the queue, which MUST be sorted, using*/
            /*an insertion sort algorithm, as insertion sort works best for */
            /*linked lists*/
            /*the currentNode will start off at the soonest tasks, so start with the */
            /*first node*/
            node * currentNode;
            currentNode = scheduledQueue->first;
            unsigned long newNodesWaitTime = getInitialWaitTime(newNode->data);
            /*before we start iterating through the queue, check to see if the new*/
            /*node can just go at the front*/
            if (newNodesWaitTime < getInitialWaitTime(currentNode->data))
            {
                /*it is less than the first node, we can enqueue it at the front*/
                /*before we do that though, we need to allocate some memory for */
                /*the task inside the newNode*/
                frontNodeEnqueue(scheduledQueue,newNode);
            }
            else
            {
                /*TODO: OPTIMIZE THIS, the if (curretNode->next) check can be moved*/
				
                /*it's not at the front, so for all the rest of the checks, check */
                /*the currentNode's next node for that data*/

                /*while we haven't found the correct location to insert the node,*/
                /*traverse the linked list in the queue, starting at the front,*/
                /*where the values are lowest, and move up*/
                while(1)
                {
                    /*first check the current node's next node to make sure it exists*/
                    /*if it doesn't we are at the end of the list and can simply append */
                    /*the node and break*/
                    if(currentNode)
                    {
                        /*the next node isn't null pointer, so we're not at the end of */
                        /*the queue, so do normal checking routine*/
                        if(newNodesWaitTime <= getInitialWaitTime(currentNode->data))
                        {
                            /*the new node goes before or at the next node in the */
                            /*chain, so add it into the link here*/
                            insertNextNode(currentNode,newNode);
                            /*also update the size, as insertNext node isn't able to*/
                            /*do that for us*/
                            (scheduledQueue->size)++;
                            break;
                        }
                        else
                        {
                            /*the node goes after this one, so iterate through the */
                            /*link by updating the currentNode reference*/
                            currentNode = currentNode->previous;
                            /*note we don't need to free any memory here, the */
                            /*currentNode reference doesn't actually constitute the*/
                            /*only reference to memory, there's still the chain*/
                            /*itself*/
                        }
                    }
                    else
                    {
                        /*the next node is a null pointer, so we are at the end of the */
                        /*queue, and we can simply normally enqueue the task at the */
                        /*end of the queue as we would normally*/
                        taskQueueNodeEnqueue(scheduledQueue,newNode);
                        break;
                    }
                }
            }
            /*now that the newNode has been put inside the queue, we can safely*/
            /*delete the newNode pointer we made and also free it*/
            /*note also that we don't safely delete the newTask pointer, that */
            /*responsibility is on the user*/
            safeDeleteNode(newNode);
            free(newNode);
            /*note we don't need to update the node's length, as it would have */
            /*been updated by one of the functions that was called to actually */
            /*insert the node*/
        }
    }
}





/** destroyQueue - deallocates all memory associated with a queue
 * 
 * @param theQueue - the queue to destroy
 */
void destroyQueue(queue * theQueue)
{
    /*first check to see if there are actually any nodes in the queue,*/
    /*there aren't any, then we don't have to go through all the nodes*/
    if (taskQueueEmpty(theQueue))
    {

        /*start at the top or front of the queue*/
        node * thisNode;
        node * nextNode;
        thisNode = theQueue->first;
        nextNode = (theQueue->first)->previous;
        /*infinite loop that can only be broken out of once all nodes have*/
        /*been deleted, */
        while (1)
        {
            /*first delete the first node*/
            safeDeleteNode(thisNode);
            free(thisNode);
            /*then update the thisNode and the nextNode*/
            thisNode = nextNode;
            /*note that once we reach the end of the queue, nextNode will itself*/
            /*be zero, so we have to check for that, if it is, we can't access*/
            /*next node*/
            if (thisNode)
            {
                /*the previous node does exist*/
                nextNode = nextNode->previous;
            }
            else
            {
                /*the next node does not exist, so just break, as we are at the*/
                /*end of the list*/
                break;
            }
        }
        
    }
    /*finally, we have deleted all the nodes, and both firstNode and nextNode*/
    /*reference null pointers, so we don't have to delete those, but we */
    /*still have the queue to take care of, so free the memory for that*/
    free(theQueue);
}


inline byte scheduledTaskReady(queue * theQueue)
{
    /*first check if the queue is empty, if it is then we automatically return 0*/
    /*if that check works, then check the initial wait time of the first task,*/
    /*if that is less than 2 milliseconds, then it is ready to run,*/
    /*if either of the conditions are false, then the result is 0 and there is*/
    /*not a scheduled task ready*/
    return (taskQueueEmpty(theQueue) && 
            (getInitialWaitTime((theQueue->first)->data)<2));
}


/** scheduledTaskQueueUpdate - update all the individual times before tasks 
 *      should be run
 * 
 * @param scheduledQueue - the queue of tasks to update
 */
void scheduledTaskQueueUpdate(queue * scheduledQueue)
{
    /*save this as a local variable to prevent having to call the function multiple times*/
	byte scheduledQueueSize = taskQueueEmpty(scheduledQueue);
	/*first check if the task queue is empty, if it is we don't need to do */
    /*anything*/
    if(scheduledQueueSize)
    {
        /*queue isn't empty, so now we check to see if it is a one node queue*/
        if (scheduledQueueSize == 1)
        {
            /*it is of size 1, so there is only the first node to update*/
            taskUpdate((scheduledQueue->first)->data);
            return;
        }
        else if (scheduledQueueSize == 2)
        {
            /*only have to update the first and last nodes*/
            taskUpdate((scheduledQueue->first)->data);
            taskUpdate((scheduledQueue->last)->data);
        }
        else
        {
            /*normal case, have to iterate through the doubly linked list*/
            /*first get a handle to the first node in the list*/
            node * firstNode = scheduledQueue->first;
            /*while there is a previous one, update the current one*/
            /*note that we are technically traversing the list backwards, hence*/
            /*the previous*/
            while(firstNode->previous)
            {
                taskUpdate(firstNode->data);
                /*update the iteration*/
                firstNode = firstNode->previous;
            }
        }
    }
    else
    {
        /*queue is empty and we can quit*/
        return;
    }
}


/** deleteFunctionFromQueue - deletes a function call of the given inside the 
 *      given queue.
 * 
 *  Note: returns a 0 if not found in the queue, a 1 if it was
 * 
 * @param theQueue - the queue to search for
 * @param ID - the ID of the function to delete
 * @return 1 if successful, 0 if not found
 */
byte deleteFunctionFromQueue(queue * theQueue, byte ID)
{
	/*save */
    /*first check if the queue is empty*/
    if(taskQueueEmpty(theQueue))
    {
        /*the queue isn't empty, so we have to search it*/
        node * tempNode;
        /*first though check if the length is 1 or 2, then we can more*/
        /*more efficiently check for the function call*/
        if (taskQueueEmpty(theQueue) == 1)
        {
            /*there is only one node in the queue, so just check that one*/
            if (getID((theQueue->first)->data) == ID)
            {
                /*we found it, delete it from the queue*/
                /*to delete the first pointer, we need to first get a temp*/
                /*pointer to the first node*/
                tempNode = theQueue->first;
                /*now update the new first pointer so that it is a null pointer*/
                (theQueue->first)->next = 0;
                /*now we can safely delete the node*/
                safeDeleteNode(tempNode);
                free(tempNode);
                /*also decrement the size of the queue*/
                theQueue->size--;
                /*we found it, so return 1*/
                return 1;
            }
            else
            {
                /*there is only one node, and we didn't find it there, so */
                /*return 0*/
                return 0;
            }
        }
        else if (taskQueueEmpty(theQueue) == 2)
        {
            if (getID((theQueue->first)->data) == ID)
            {
                /*we found it, delete it from the queue*/
                /*to delete the first pointer, we need to first get a temp*/
                /*pointer to the first node*/
                tempNode = theQueue->first;
                /*now point the queue's first pointer to the second node*/
                theQueue->first = tempNode->previous;
                /*now update the new first pointer so that it is a null pointer*/
                (theQueue->first)->next = 0;
                /*now we can safely delete the node*/
                safeDeleteNode(tempNode);
                free(tempNode);
                /*also decrement the size of the queue*/
                theQueue->size--;
                /*we found it, so return 1*/
                return 1;
            }
            else if (getID((theQueue->last)->data) == ID)
            {
                /*we found it at the end, so delete it from the queue*/
                /*to delete the last node, we first have to set get a temp*/
                /*pointer to the last node*/
                tempNode = theQueue->last;
                /*now point the queue's last pointer at the second to last*/
                /*node*/
                theQueue->last = tempNode->next;
                /*now update the new last pointer so that it is a null pointer*/
                (theQueue->last)->previous = 0;
                /*now we can safely delete the node*/
                safeDeleteNode(tempNode);
                free(tempNode);
                /*also decrement the size of the queue*/
                theQueue->size--;
                /*return 1 because we found it*/
                return 1;
            }
            else
            {
                /*we didn't find it in the only two nodes, the first and last */
                /*nodes in the queue, so return 0*/
                return 0;
            }
        }
        else
        {
            /*normal base case of more than 2 nodes, so we have to perform a */
            /*linear search for the item*/
            /*start at the front of the linked list*/
            tempNode = theQueue->first;
            /*while there is a next node*/
            while(tempNode->previous)
            {
                /*check the current node*/
                if(getID(tempNode->data) == ID)
                {
                    /*we found it, so delete it and return 1*/
                    
                    /*NOTE: THIS CAN BE OPTIMIZED BY COMBINING THIS CHECK AT THE*/
                    /*TOP OF THE FUNCTION WITH THE OTHER CASES IN WHICH WE CHECK*/
                    /*THE FIRST NODE*/
                    
                    /*this is the general delete case*/
                    /*first, if the tempNode's previous pointer exists, point the */
                    /*tempNode's next node at tempNode's previous node*/
                    if (tempNode->next)
                    {
                        (tempNode->next)->previous = tempNode->previous;
                    }
                    else
                    {
                        /*now that we know that this node has to be the*/
                        /*first node, we also have to repoint the first node*/
                        /*pointer in the queue*/
                        theQueue->first = tempNode->previous;
                    }
                    /*next point tempNode's previous node at tempNode's next */
                    /*node, we don't have to check it, it either exists or is a*/
                    /*null pointer, in which case we don't really have to worry*/
                    (tempNode->previous)->next = tempNode->next;
                    /*we can now safely free the node*/
                    safeDeleteNode(tempNode);
                    free(tempNode);
                    /*also decrement the size of the queue*/
                    theQueue->size--;
                    /*return 1 because we found it*/
                    return 1;
                }
                else
                {
                    /*increment the tempNode pointer to the next one*/
                    tempNode = tempNode->previous;
                }
            }
            /*note that at the end of the while loop, tempNode's next pointer is*/
            /*zero, but tempNode could still be the one we are looking for, so*/
            /*check it anyways now*/
            if (getID(tempNode->data) == ID)
            {
                /*we found it at last!*/
                /*we know that the previous pointer of this is null, so we don't*/
                /*need to update that, but we do have to worry about this being*/
                /*the last node in the queue, in which case we have to repoint*/
                /*the last node pointer inside the queue*/
                theQueue->last = tempNode->next;
                /*next null the pointer of the new last node*/
                (theQueue->last)->previous = 0;
                /*now we can delete the node and decrement the size*/
                safeDeleteNode(tempNode);
                free(tempNode);
                /*also decrement the size of the queue*/
                theQueue->size--;
                /*return 1 because we found it*/
                return 1;
            }
            /*if we exit the while loop, it means we got all the way through the*/
            /*queue without finding the ID, so return 0 as we didn't find it*/
            return 0;
        }
    }
    else
    {
        /*the queue is empty, so the function call wasn't found*/
        return 0;
    }
}


}




#endif	/* QUEUE_H */






task * nextTask;
task * immediateTaskToRun;
task * scheduledTaskToRun;
queue * immediateTaskQueue;
queue * scheduledQueue;
byte * pinConfigurations;



inline void bootPrefunctions()
{
    /*GENERATED CODE THAT QUEUES UP TASK AT BOOT FOLLOWS*/
    `preTaskSetup`
    /*END GENERATED CODE THAT QUEUES UP TASK AT BOOT*/
}


void setup()
{
    immediateTaskQueue = createQueue();
    scheduledQueue = createQueue();  
    pinConfigurations = (byte *)calloc(3,sizeof(byte));
    SerialLink.begin(AVR_BAUD_RATE);
#if defined (ARDUINO_AVR_YUN)
    //begin the bridge library
    Serial1.begin(ATHEROS_BAUD_RATE);
#endif
}

void loop()
{
  /*main event loop*/

    bootPrefunctions();

    while(1)
    {
		/*first need to check the serial buffer*/
		
		if(SerialLink.available())
		{
			/*there is data off of the serial buffer*/
			/*save the data inside a new task*/
			nextTask = getSerialFirmataTask();
			/*check to make sure the pointer isn't null,*/
			/*it could be if there was a corrupted message on the serial buffer*/
			/*or a non task message was sent somehow*/
			if (!nextTask)
			{
				/*continue to the next while loop iteration, it may take a few*/
				/*loops through to clear the junk bytes off of the serial buffer, */
				/*this will ensure that we don't actually continue to the rest of*/
				/*the while loop without a valid task to handle*/
				continue;
			}
			switch(getType(nextTask))
			{
				case CONFIG:
					/*run the config task*/
					configTaskRun(nextTask,pinConfigurations);
					/*we can now safely deallocate the nextTask memory pointer*/
					safeDeleteTask(nextTask);
					break;
				case ANALOG_READ:
				case DIGITAL_READ:
					/*run the read task*/
					readTaskRun(nextTask,pinConfigurations);
					/*we can now safely deallocate the nextTask memory pointer*/
					safeDeleteTask(nextTask);
					break;
				case ANALOG_WRITE:
				case DIGITAL_WRITE:
					/*run the write task*/
					writeTaskRun(nextTask,pinConfigurations);
					/*we can now safely deallocate the nextTask memory pointer*/
					safeDeleteTask(nextTask);
					break;
				case FUNCTION_CALL:
					/*switch on what type of function it is*/
					if (getInitialWaitTime(nextTask))
					{
						/*the wait time is more than 0 seconds, so */
						/*add it to the scheduling queue*/
						taskScheduledQueueAdd(scheduledQueue,
							nextTask);
						/*now delete the task*/
						safeDeleteTask(nextTask);
					}
					else
					{
						/*the function should be handled */
						/*immediately, so put it in the queue*/
						taskQueueEnqueue(immediateTaskQueue,nextTask);
						/*now that the nextTask has been stored,*/
						/*we can safely free that memory*/
						safeDeleteTask(nextTask);
						/*now dequeue the front task from the immediate*/
						/*function queue*/
						immediateTaskToRun = taskQueueDequeue(
							immediateTaskQueue);
						/*now run the task and enter that subroutine*/
						runTask(immediateTaskToRun,pinConfigurations);
						/*we have finished running the task, so we*/
						/*can safely free that memory*/
						safeDeleteTask(immediateTaskToRun);
					}
					break;
				case FUNCTION_DELETE:
					/*first try deleting the function from the scheduledQueue*/
					if(deleteFunctionFromQueue(scheduledQueue,getID(nextTask))==0)
					{
						/*deleteFunctionFromQueue returns a 0 if the function wasn't found in the queue,*/
						/*so we need to look at the immediate queue*/
						deleteFunctionFromQueue(immediateTaskQueue,getID(nextTask));
					}
					/*now free the task*/
					safeDeleteTask(nextTask);
					break;
				default:
					/*there is nothing to do, this means there was an*/
					/*error from the serial buffer to get a task other*/
					/*than the above*/
					break;
			}
			/*now that task off of the serial buffer has been handled,*/
			/*we need to update the elapsed time for the schduled */
			/*functions*/
			do
			{
				/*while there is a task ready to run, update the task*/
				/*queue, and check if there is a new task ready to*/
				/*run*/
				scheduledTaskQueueUpdate(scheduledQueue);
				/*now check if there are any new functions off that queue*/
				/*that are ready*/
				if(scheduledTaskReady(scheduledQueue))
				{
					/*there is at least one ready to run, so grab it and */
					/*run it now*/
					scheduledTaskToRun = taskQueueDequeue(
						scheduledQueue);
					runTask(scheduledTaskToRun,pinConfigurations);
					/*that task has now finished, so we can safely free*/
					/*that memory*/
					safeDeleteTask(scheduledTaskToRun);
				}
			} while(scheduledTaskReady(scheduledQueue));
		}/*end SerialLink.available if*/
		else
		{
			/*there isn't anything new in the serial buffer*/
			/*therefore, we should check the scheduled tasks, and see*/
			/*if any of those are ready*/
			if (scheduledTaskReady(scheduledQueue))
			{
				/*there is a scheduled task ready to run, so grab it*/
				/*off the queue and run it*/
				scheduledTaskToRun = taskQueueDequeue(
					scheduledQueue);
				runTask(scheduledTaskToRun,pinConfigurations);
				/*that task has now finished running, so we can safely */
				/*free that memory*/
				safeDeleteTask(scheduledTaskToRun);
			}
			/*there aren't any scheduled tasks ready, check to see*/
			/*if there are any immeadiate tasks ready from that */
			/*queue*/
			else if(taskQueueEmpty(immediateTaskQueue))
			{
				/*there is a task in the immediate queue to run*/
				/*grab it and run it*/
				immediateTaskToRun = taskQueueDequeue(
					immediateTaskQueue);
				runTask(immediateTaskToRun,pinConfigurations);
				/*now that we have run that task, we can */
				/*safely free that memory*/
				safeDeleteTask(immediateTaskToRun);
			}
			/*regardless of which type of task we ran or didn't run,*/
			/*now we enter back into the do while loop checking for */
			/*scheduled tasks to be ready*/
			do
			{
				/*while there is a task ready to run, update the task*/
				/*queue, and check if there is a new task ready to*/
				/*run*/
				scheduledTaskQueueUpdate(scheduledQueue);
				/*now check if there are any new functions off that queue*/
				/*that are ready*/
				if(scheduledTaskReady(scheduledQueue))
				{
					/*there is at least one ready to run, so grab it and */
					/*run it now*/
					scheduledTaskToRun = taskQueueDequeue(
						scheduledQueue);
					runTask(scheduledTaskToRun,pinConfigurations);
					/*that task has now finished, so we can safely free*/
					/*that memory*/
					safeDeleteTask(scheduledTaskToRun);
				}
			} while(scheduledTaskReady(scheduledQueue));
		}/*end else block for no SerialLink.available()*/
      }/*end infinite while loop*/
}
