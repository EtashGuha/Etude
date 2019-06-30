(* ::Package:: *)

(* Paclet Info File *)

(* created 2015.01.23*)

Paclet[
  Name -> "DeviceDriver_Vernier",
  Version -> "1.1.2",
  MathematicaVersion -> "10+",
  Creator ->"Armeen Mahdian <armeenm@wolfram.com>, Ian Johnson <ijohnson@wolfram.com>",
  Loading->Automatic,
  Extensions -> {
  	{"LibraryLink",SystemID->{"MacOSX-x86-64", "Linux", "Linux-ARM", "Linux-x86-64", "Windows", "Windows-x86-64"}},
  	{"Kernel",
  		Root -> "Kernel",
  		Context -> {"VernierLink`"}
  	},
    {"Resource",
    	Root->"Resources",
    	Resources->{
	    	{"Logo","Bitmaps/vernier_logo.jpg"}
    	}
    }
}]
