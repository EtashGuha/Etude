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

#include "stepper28BYJ48ATtiny85.h"
#include <avr/io.h>
#include <util/delay.h>

uint8_t adafruitTrinketDDRMapping(uint8_t pin)
{
    switch (pin) {
        case 0:
            return DDB0;
        case 1:
            return DDB1;
        case 2:
            return DDB2;
        case 3:
            return DDB3;
        case 4:
            return DDB4;
        default:
            return pin;
    }
}

uint8_t adafruitTrinketPORTMapping(uint8_t pin)
{
    switch (pin) {
        case 0:
            return PORTB0;
        case 1:
            return PORTB1;
        case 2:
            return PORTB2;
        case 3:
            return PORTB3;
        case 4:
            return PORTB4;
        default:
            return pin;
    }
}

uint8_t switchingSequences[8][4] = {
    1, 0, 0, 0,
    1, 1, 0, 0,
    0, 1, 0, 0,
    0, 1, 1, 0,
    0, 0, 1, 0,
    0, 0, 1, 1,
    0, 0, 0, 1,
    1, 0, 0, 1
};

// general utilites
void pinWrite(uint8_t val, uint8_t port)
{
    val?(PORTB |= 1 << port):(PORTB &= ~(1 << port));
}

// motor utilites
void StepperMotor::writeSequence(uint8_t i)
{
    for(int j = 0; j < 4; j++)
    {
        pinWrite(switchingSequences[i][j], motorPinPort[j]);
    }
}

void StepperMotor::rotateClockwise()
{
    for(int i = 7; i >= 0; i--)
    {
        writeSequence(i);
        _delay_us(1200);
    }
}

void StepperMotor::rotateAntiClockwise()
{
    for(int i = 0; i < 8; i++)
    {
        writeSequence(i);
        _delay_us(1200);
    }
}

//public
StepperMotor::StepperMotor(uint8_t pins[4])
{
    for(int i = 0; i < 4; i++)
    {
        DDRB |= (1 << adafruitTrinketDDRMapping(pins[i]));
        this->motorPinPort[i] = adafruitTrinketPORTMapping(pins[i]);
    }
}

void StepperMotor::rotate(int steps)
{
    if (steps == 0)
    {
        return;
    }
    int c = 0;
    if(steps > 0)
    {
        while (c < steps)
        {
            rotateClockwise();
            c++;
        }
    }
    else
    {
        while (c > steps)
        {
            rotateAntiClockwise();
            c--;
        }
    }
    return;
}
