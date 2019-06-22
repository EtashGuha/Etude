# Encoder Function API

This document describes the API that encoder functions must conform to. 

NOTE: This document is a work-in-progress.

# Input

ToEncoderFuntion property should be a function that takes the parameters association of the encoder as #1 and returns a function to actually do the encoding, called the *encoder function*.

## Input

The encoder function should expect one argument, which will be a batch of user data to encode. This batch will be of size one for a net that is being run on a single example. For training, it will be the size of a training batch.

The individual values in the batch (where possible), are NumericArrays. If those individual values are scalars then NumericArrays cannot be used of course.

## Output

The encoder function should return one of the following two cases:

* A NumericArray or list of NumericArrays (of any type)

If a single numeric array is returned, it should have first dimension equal to the batch size (the number of inputs), *obviously*.

## Failure

NOTE: this is not supported yet, but is left here as an aspirational note. Currently, the API is that EncodeFail is called with a message.

If an encoder function fails because the input is incorrect, it should return a special value to indicate this, which can take the following three forms:

* $Failed -- the input was invalid, but we don't know which batch element was invalid
* $Failed["reason"] -- there was a specific problem, e.g. file did not exist, was corrupt, etc. not a generic message.

## Error Messages

NetTrain::inbatchdata Input #3 was invalid: file 'foo.jpeg' did not exist. The batch containing this input will be skipped.
...
...
...
NetTrain::skippedbatchs: 23424 examples were skipped because of invalid data. See the InvalidInputs property for more information. 

"InvalidInputs" -> <|3 -> "file 'foo' did not exist', 5 -> 'file was 'foo' was corrupt', 32 -> ...|>








