(* ::Package:: *)

(* Paclet Info File *)

(* created 2015.01.23*)

Paclet[
  Name -> "DeviceDriver_Arduino",
  Version -> "1.1.2",
  MathematicaVersion -> "10+",
  Creator ->"Ian Johnson <ijohnson@wolfram.com>",
  Loading->Automatic,
  Internal->True,
  
  Extensions -> {
  	{"Kernel",
  		Root -> "Kernel",
  		Context -> 
  			{
	  			"ArduinoLink`",
	  			"ArduinoCompile`",
	  			"ArduinoUpload`",
	  			"AVRCCompiler`",
	  			"SketchTemplate`",
	  			"Firmata`"
  			}
  	},
    {"Documentation",
		MainPage->"ReferencePages/Devices/Arduino",
		Language->All
	},
    {"Resource",
    	Root->"Resources",
    	Resources->{
	    	{"Sketch","CSource/SketchTemplate.cpp"},
	    	{"Logo","Bitmap/community_logo.png"},
	    	{"ArduinoLibraries","CSource/libraries"},
	    	{"ArduinoCores","CSource/cores/arduino"},
	    	{"ArduinoVariants","CSource/variants"},
	    	{"avr-libc-include","CSource/avr-libc/1.8.1/avr/include"},
	    	{"avr-libc-lib","CSource/avr-libc/1.8.1/avr/lib"}
    	}
    }
}]
