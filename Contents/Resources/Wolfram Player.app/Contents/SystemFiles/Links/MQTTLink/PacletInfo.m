(* ::Package:: *)

(* Paclet Info File *)

(* created 2015.01.23*)

Paclet[
  Name -> "MQTTLink",
  Version -> "1.1.7",
  MathematicaVersion -> "10+",
  Creator ->"Ian Johnson <ijohnson@wolfram.com>",
  Loading->Automatic,
  Internal->True,
  Extensions -> {
  	{"LibraryLink",SystemID->{"MacOSX-x86-64", "Linux", "Linux-ARM", "Linux-x86-64", "Windows", "Windows-x86-64"}},
  	{"Kernel",
  		Root -> "Kernel",
  		Context -> {"MQTTLink`"}
  	},
    {"Resource",
    	Root->"Resources",
    	Resources->{
	    	{"ClientImage","Bitmaps/client_image.png"},
	    	{"Binaries","Binaries"},
	    	{"BrokerConfig","config.txt"}
    	}
    }
}]
