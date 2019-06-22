/**********************************************************************************************
 
 Created by :    Suba Thomas, Wolfram Research Inc.
 Date       :    May 31, 2017
 
 This stepper28BYJ48ATtiny85 library is provided for demonstrative and example purposes only.
 It is made available "as is", without warranty of any kind, express or implied. In no event
 shall Wolfram Research, Inc. or the individual authors of the library, be held liable in any
 way arising from, out of or in connection with the library.
 
 The full documentation for the open-loop control of the 28BYJ48 stepper motor is available at
 paclet:MicrocontrollerKit/workflow/StepperMotorControl
 or
 https://reference.wolfram.com/language/MicrocontrollerKit/workflow/StepperMotorControl.html
 
 **********************************************************************************************/

#include <stdint.h>

class StepperMotor
{
    private:
        uint8_t motorPinPort[4];
        void writeSequence(uint8_t i);
        void rotateClockwise();
        void rotateAntiClockwise();
    public:
        StepperMotor(uint8_t pins[4]);
        void rotate(int steps);
    
};
