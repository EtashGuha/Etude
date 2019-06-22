(* ::Package:: *)

(* Wolfram Language Package *)

(*==========================================================================================================
			
					ARDUINO COMPILE
			
Author: Ian Johnson
			
Copyright (c) 2015 Wolfram Research. All rights reserved.			


Arduino Compile is a package to take a valid C/C++ file and produce a compiled .o file for subsequent upload
to an Arduino.

CURRENT SUPPORTED BOARDS:
~Arduino Uno

USER ACCESSIBLE FUNCTIONS:
arduinoCompile

==========================================================================================================*)



BeginPackage["ArduinoCompile`",{"ArduinoLink`"}]
(* Exported symbols added here with SymbolName::usage *)  

arduinoCompile::usage="arduinoCompile takes the location c/c++ program in a text file and will compile it in this location. It returns the output from the compilation, as well as the "



Begin["`Private`"] (* Begin Private Context *) 


$thisFileDir = DirectoryName@$InputFileName

Get[FileNameJoin[{$thisFileDir,"AVRCCompiler.m"}]]

Needs[ "CCompilerDriver`"];
Needs["CCompilerDriver`CCompilerDriverBase`"];




(*==========================================================================================================
arduinoCompile will compile the c/c++ file at the given location. The directory that the file is located
in is taken to be the folder where compilation output is put. No output files in this directory are deleted.

The arduino install directory must be passed to this function, it is not the responsibility of the compiler
to find the install location.

arduinoCompile uses CreateObjectFile from CCompilerDriver and the custom AVRCCompiler file to compile the
program and libraries with the avr-gcc and avr-g++ compilers found in the Arduino software.



algorithm for compiling:

first get all the libraries in order. Need to have a list of the library locations

then use CreateObjectFile with g++ to create the output file of the SketchTemplate.cpp

then use CreateObjectFile with g++ to create the output file of the libfile.cpp for all the libraries(if necessary)

then use CreateObjectFile with gcc to create the output file of the utillibfile.c for all the libraries(if necessary)

finally, compile all these together to produce an .elf file






PARAMETERS:
	arduinoInstallLocation - the location of Arduino software
	buildLocation - the location of the temp folder to look for files and the firmware file
	fileName - the name of the file to be compiled

RETURNS: 
	association of input command to output from each command, if fully successful
	$Failed, if any part of the compilation process fails


OPTIONS:
	"Debug" - whether or not debugging info should be output
	"CleanIntermediate" - whether or not to clean up all the intermediate files, like the object files and such
	"ArduinoVersion" - the version of the arduino software to define when compiling it
	"AVRGCCLocation" - the location to find the avr-gcc utilities (setting this to default causes it to use the ones that are supposed to be in the arduino install location)
	"StandardArduinoLibrariesLocation" - the location to find all the standard arduino libraries that need to also be compiled


==========================================================================================================*)

Options[arduinoCompile]=
	{
		"Debug"->False,
		"CleanIntermediate"->True,
		"ArduinoVersion"->"10607",
		"AVRGCCLocation"->Default,
		"StandardArduinoLibrariesLocation"->Default,
		"ChipPartNumber"->Default,
		"ArduinoBoardDefine"->Default,
		"ExtraDefines"->{},
		"ChipArchitecture"->"avr5"
	};

arduinoCompile::filenotfound="Unable to import `1` as program text file";
arduinoCompile::mainfail="Compilation of the sketch failed in the first stage";
arduinoCompile::execFailed="Compilation of the sketch failed in the last stage";
arduinoCompile::cLibraryFail="Compilation of the C library `1` failed";
arduinoCompile::cppLibraryFail="Compilation of the C++ library `1` failed";
arduinoCompile::arduinoLibraryFail="Compilation of the Arduino standard library `1` failed"

arduinoCompile[arduinoInstallLocation_,buildLocation_,fileName_,OptionsPattern[]]:=Module[
	{
		(*constant strings for CCompilerDriver, avr-gcc, and avr-g++*)
		highVerbose= "-v -v -v -v",
		lowVerbose = "-v",
		OptimizeForSize = "-Os",
		allWarnings = "-Wall",
		noExceptions = "-fno-exceptions",
		functionSectioning = "-ffunction-sections",
		dataSectioning = "-fdata-sections",
		noThreading="-fno-threadsafe-statics",
		microcontrollerUnitSpec = "-mmcu=",
		arduinoChipNumber = If[OptionValue["ChipPartNumber"]===Default||Not[StringQ[OptionValue["ChipPartNumber"]]],
			(*THEN*)
			(*use default for Arduino Uno is atmega328p*)
			"atmega328p",
			(*ELSE*)
			(*use whatever was passed in*)
			OptionValue["ChipPartNumber"]
		],
		clockCrystalSpeedDef = "F_CPU=",
		clockSpeed ="16000000L",
		onlyHeaderFileOutput ="-MMD",
		arduinoVersionDef = "ARDUINO=",
		(*default to 10603 if the passed in option is anything other than a string*)
		arduinoVersion = If[StringQ[OptionValue["ArduinoVersion"]],OptionValue["ArduinoVersion"],"10607"],
		linkerDirectiveOnlyUsedCode = "--gc-sections",
		linkerDirectiveVerboseOutput="--verbose",
		arduinoBoard=If[OptionValue["ArduinoBoardDefine"]===Default||Not[StringQ[OptionValue["ArduinoBoardDefine"]]],
			(*THEN*)
			(*use default for Arduino Uno is atmega328p*)
			"ARDUINO_AVR_UNO",
			(*ELSE*)
			(*use whatever was passed in*)
			OptionValue["ArduinoBoardDefine"]
		],
		originalPathEnvironment=Environment["PATH"],
		debugNativeOutput="-g",
		extraCompileDefines,
		compilerLocation,
		arduinoStandardLibraryDirectories,
		(*this is to let us tell avr-gcc where to find libc and libm, as we want it to use our version, not anyone else's*)
		noStandardCLibraries="-nostdinc",
		avrGCCIncludeLocation,
		languageSpec="-x",
		assembly="assembler-with-cpp",
		(*this is so that when we compile the final executable, we can specify where to find the .crt files, the files that contain init binary code location stuff*)
		(*by default avr-gcc tries to use it's own (but on some platforms it doesn't have any), and we want to use ours with avr-libc in PacletResources*)
		noStandartStartFiles="-nostartfiles",
		gnuCPP11Std="-std=gnu++11",
		gnuC11Std="-std=gnu11"
	},
	(
		extraCompileDefines=With[{optionVal=OptionValue["ExtraDefines"]},
			Which[
				Head[optionVal]===List,(*it's a list, so just check that the list is of all strings*)
				(
					If[AllTrue[StringQ][optionVal],
						(*THEN*)
						(*the defines are all strings and we can use them*)
						StringTrim/@optionVal,
						(*ELSE*)
						(*the defines aren't all strings, don't use any of them*)
						{}
					]
				),
				Head[optionVal]===Association,(*it's an associaiton of defines -> values*)
				(
					With[{keys=Keys[optionVal],vals=Values[optionVal]},
						If[AllTrue[StringQ][Join[keys,vals]],
							(*THEN*)
							(*all the keys and values are valid*)
							(*to make them valid defines, string trim the keys and values, insert an "=", and string join them*)
							(*with an association like <|"a"->"b","c"->"d","e"->"f"|>, this produces {"a=b","c=d","e=f"}*)
							StringJoin[StringTrim/@Riffle[#,"="]]&/@Transpose[{keys, vals}],
							(*ELSE*)
							(*some are wrong, don't use any*)
							{}
						]
					]
				),
				True, (*all other cases just use empty list*)
				{}
			]
		];
		compilerLocation = If[OptionValue["AVRGCCLocation"]===Default,
			(*THEN*)
			(*use the default directory path inside the arduino install location*)
			FileNameJoin[{arduinoInstallLocation,"hardware","tools","avr","bin"}],
			(*ELSE*)
			(*a different location was specified, try and use that one instead*)
			(
				If[FileExistsQ[OptionValue["AVRGCCLocation"]],
					(*THEN*)
					(*the location specified exists, so use that one instead*)
					OptionValue["AVRGCCLocation"],
					(*ELSE*)
					(*the location specified was incorrect, so try using default*)
					FileNameJoin[{arduinoInstallLocation,"hardware","tools","avr","bin"}]
				]
			)
		];
		
		arduinoStandardLibraryDirectories = OptionValue["StandardArduinoLibrariesLocation"];
		
		avrGCCIncludeLocation=FileNameJoin[
			{
				FileNameDrop[compilerLocation],
				"lib",
				"gcc",
				"avr",
				(*this figures out the version number of gcc*)
				FileNameTake[First[FileNames["*",FileNameJoin[{FileNameDrop[compilerLocation],"lib","gcc","avr"}]]]],
				"include"
			}
		];
		
		(*first add the compiler location to the environment path so the commands can find the cygwin dll on windows for version 1.6.0 of arduino*)
		If[$OperatingSystem === "Windows",
			SetEnvironment["PATH"->originalPathEnvironment<>";"<>arduinoInstallLocation];
		];
		
		(*reset $output for returning at the end*)
		$output=<||>;
		
		(*get a list of the libraries using libFinder function*)
		libs = libFinder[buildLocation];
		
		programText = Import[FileNameJoin[{buildLocation,fileName}],"Text"];
		
		(*if the file isn't found, issue message and return $Failed*)
		If[programText === $Failed, 
			Message[arduinoCompile::filenotfound,FileNameJoin[{buildLocation,fileName}]];
			Return[$Failed]
		];
		
		If[OptionValue["Debug"]===True,
			(
				Print["Starting compilation..."];
				$startCompileTime = AbsoluteTime[];
			)
		];

		If[ libs === {},
			(*THEN*)
			(*we don't have libraries, so we can just jump to the final compilation*)
			(
				If[OptionValue["Debug"]===True,
					(
						Print["No libraries found by arduinoCompile"];
					)
				];
				
				(*compile the main program file first*)
				CreateObjectFile[
					programText,
					fileName,
					"Compiler"->AVRCCompiler,
					"CompilerInstallation"->compilerLocation,
					"CompilerName"->"avr-g++",
					"TargetDirectory"->buildLocation,
					"CompileOptions"->CommandJoin[
						Riffle[
							{
								highVerbose,
								OptimizeForSize,
								debugNativeOutput,
								allWarnings,
								noExceptions,
								functionSectioning,
								dataSectioning,
								noThreading,
								onlyHeaderFileOutput,
								noStandardCLibraries,
								microcontrollerUnitSpec<>arduinoChipNumber,
								gnuCPP11Std
							},
							" "]
						],
					"Defines"->
					Join[
						{
							"SERIAL_RX_BUFFER_SIZE="<>"128",
							"SERIAL_TX_BUFFER_SIZE="<>"64",
							clockCrystalSpeedDef<>clockSpeed,
							arduinoBoard,
							arduinoVersionDef<>arduinoVersion
						},
						extraCompileDefines
					],
					"LibraryDirectories" -> arduinoStandardLibraryDirectories,
					"IncludeDirectories" -> Join[{avrGCCIncludeLocation,PacletResource["DeviceDriver_Arduino","avr-libc-include"]},arduinoStandardLibraryDirectories],
					"ShellOutputFunction"->(compilationProgressCatcher[#,"output"]&),
					"ShellCommandFunction"->(compilationProgressCatcher[#,"command","Debug"->OptionValue["Debug"]]&),
					"CleanIntermediate"->False,
					(*SystemIncludeDirectories, SystemLibraryDirectories, and SystemLibraries are for
				  	Wolfram C Libraries, so they can be discarded*)
					"SystemIncludeDirectories" -> {}
				];
				
				(*increment the progress indiciator for the progress bar*)
				ArduinoLink`Private`$compilationProgressIndicator++;
				
				(*THE FOLLOWING IS A WORKAROUND FOR A WIERD BUG WHERE CCOMPILERDRIVER THINKS IT FAILED EVEN WHEN 
				IT SUCCESSFULLY COMPLETES SO HERE WE MANUALLY CHECK TO SEE IF IT FAILED*)
				If[fileCheck[getWolfWorkDir[buildLocation],fileName<>".o"],
					(*THEN*)
					(*the file was compiled and the output exists*)
					moveFiles[getWolfWorkDir[buildLocation]],
					(*ELSE*)
					(*the file doesn't exist, so the compilation failed*)
					(*Raise Message and throw $Failed*)
					(
						Message[arduinoCompile::mainfail];
						If[$OperatingsSystem === "Windows",
							SetEnvironment["PATH"->originalPathEnvironment];
						];
						Return[$Failed]
					)
				];
				
				(*now compile the standard arduino library files*)
				$arduinoStandardLibraryFiles=compileArduinoStandardLibraries[
					arduinoInstallLocation,
					buildLocation,
					"AVRGCCLocation"->OptionValue["AVRGCCLocation"],
					"StandardArduinoLibrariesLocation"->OptionValue["StandardArduinoLibrariesLocation"],
					"ArduinoVersion"->arduinoVersion,
					"ChipPartNumber"->arduinoChipNumber,
					"ArduinoBoardDefine"->arduinoBoard,
					"ExtraCompileDefines"->extraCompileDefines
				];
				
				(*check to make sure the standard arduino libraries all compiled okay*)
				If[$arduinoStandardLibraryFiles===$Failed,
					(*THEN*)
					(*the libraries failed, but a message should already have been generated, so just reset the path environment variable and exit*)
					(
						If[$OperatingSystem === "Windows",
							SetEnvironment["PATH"->originalPathEnvironment];
						];
						Return[$Failed];
					)
				];

				CreateExecutable[
					{FileNameJoin[{buildLocation,fileName<>".o"}]},
					fileName,
					"Compiler"->AVRCCompiler,
					"CompilerInstallation"->compilerLocation,
					"CompilerName"->"avr-gcc",
					"TargetDirectory"->buildLocation,
					"CompileOptions" ->CommandJoin[
						Riffle[
							{
								lowVerbose,
								OptimizeForSize,
								allWarnings,
								noStandardCLibraries,
								noStandartStartFiles,
								microcontrollerUnitSpec<>arduinoChipNumber,
								"-Wl,"<>linkerDirectiveOnlyUsedCode<>","<>linkerDirectiveVerboseOutput,
								gnuC11Std
							},
							" "]
						],
					"ShellOutputFunction"->(compilationProgressCatcher[#,"output"]&),
					"ShellCommandFunction"->(compilationProgressCatcher[#,"command","Debug"->OptionValue["Debug"]]&),
					"CleanIntermediate" -> False,
					(*TODO: figure out way to determine avr5, rather than having it hardcoded*)
					"LibraryDirectories"->{FileNameJoin[{PacletResource["DeviceDriver_Arduino","avr-libc-lib"],avrChipPartNumberToArchitecture[arduinoChipNumber]}],buildLocation},
					(*SystemIncludeDirectories, SystemLibraryDirectories, and SystemLibraries are for
				  	Wolfram C Libraries, so they can be discarded*)
					"SystemIncludeDirectories" -> {},
					"SystemLibraryDirectories" -> {},
					"SystemLibraries" -> {},
					"ExtraObjectFiles"->Join[
						{
							FileNameJoin[
								{
									PacletResource["DeviceDriver_Arduino","avr-libc-lib"],
									avrChipPartNumberToArchitecture[arduinoChipNumber],
									avrChipNumberToCRTFile[arduinoChipNumber]
								}
							]
						},
						$arduinoStandardLibraryFiles
					],
					(*m is the math library for the avr platform*)
					"Libraries"->"m"
				];
				
				(*increment the progress indiciator for the progress bar*)
				ArduinoLink`Private`$compilationProgressIndicator++;
				
				If[fileCheck[getWolfWorkDir[buildLocation],fileName<>".elf"],
					(*THEN*)
					(*the file was compiled and the output exists*)
					moveFiles[getWolfWorkDir[buildLocation]],
					(*ELSE*)
					(*the file doesn't exist, so the compilation failed*)
					(
						(*Raise Message and return $Failed*)
						Message[arduinoCompile::execFailed];
						If[$OperatingSystem === "Windows",
							SetEnvironment["PATH"->originalPathEnvironment];
						];
						Return[$Failed]
					)
				]
			),
			(*ELSE*)
			(*we have libraries, so we need to compile those, then do the final compilation*)
			(	
				(*print off some debgging info if requested*)
				If[OptionValue["Debug"]===True,
					(
						Print["The libraries ",libs," were found by arduinoCompile"];
					)
				];
				
				
				(*compile the main program file first*)
				CreateObjectFile[
					programText,
					fileName,
					"Compiler"->AVRCCompiler,
					"CompilerInstallation"->compilerLocation,
					"CompilerName"->"avr-g++",
					"TargetDirectory"->buildLocation,
					"CompileOptions"->CommandJoin[
						Riffle[
							{
								highVerbose,
								OptimizeForSize,
								debugNativeOutput,
								allWarnings,
								noExceptions,
								functionSectioning,
								dataSectioning,
								noThreading,
								onlyHeaderFileOutput,
								noStandardCLibraries,
								microcontrollerUnitSpec<>arduinoChipNumber,
								gnuCPP11Std
							},
							" "]
						],
					"Defines"->
					Join[
						{
							"SERIAL_RX_BUFFER_SIZE="<>"128",
							"SERIAL_TX_BUFFER_SIZE="<>"64",
							clockCrystalSpeedDef<>clockSpeed,
							arduinoBoard,
							arduinoVersionDef<>arduinoVersion
						},
						extraCompileDefines
					],
					"LibraryDirectories" -> Join[arduinoStandardLibraryDirectories,{FileNameJoin[{buildLocation,"libs"}]}],
					"IncludeDirectories" -> Join[{avrGCCIncludeLocation,PacletResource["DeviceDriver_Arduino","avr-libc-include"],FileNameJoin[{buildLocation,"libs"}]},arduinoStandardLibraryDirectories],
					"ShellOutputFunction"->(compilationProgressCatcher[#,"output"]&),
					"ShellCommandFunction"->(compilationProgressCatcher[#,"command","Debug"->OptionValue["Debug"]]&),
					"CleanIntermediate"->False,
					(*SystemIncludeDirectories, SystemLibraryDirectories, and SystemLibraries are for
				  	Wolfram C Libraries, so they can be discarded*)
		  			"SystemIncludeDirectories" -> {}
				];
				
				If[fileCheck[getWolfWorkDir[buildLocation],fileName<>".o"],
					(*THEN*)
					(*the file was compiled and the output exists*)
					moveFiles[getWolfWorkDir[buildLocation]],
					(*ELSE*)
					(*the file doesn't exist, so the compilation failed*)
					(*Raise Message and throw $Failed*)
					(
						Message[arduinoCompile::mainfail];
						If[$OperatingSystem === "Windows",
							SetEnvironment["PATH"->originalPathEnvironment];
						];
						Return[$Failed]
					)
				];
				
				(*increment the progress indiciator for the progress bar*)
				ArduinoLink`Private`$compilationProgressIndicator++;
				
				(*for each library we found compile it with CreateOjectFile, switching on which kind of library it is*)
				Do[
					(
						(*increment the progress indiciator for the progress bar*)
						ArduinoLink`Private`$compilationProgressIndicator++;
					
						Which[
							ToLowerCase[FileExtension[lib]]==="c",
							(*the file is a c library, so have to compile it with gcc*)
							(
								If[OptionValue["Debug"],
									Print[lib," is a c library"];
								];
								CreateObjectFile[
									{FileNameJoin[{buildLocation,"libs",lib}]},
									lib,
									"Compiler"->AVRCCompiler,
									"CompilerInstallation"->compilerLocation,
									"CompilerName"->"avr-gcc",
									"TargetDirectory"->buildLocation,
									"CompileOptions"->CommandJoin[
										Riffle[
											{
												highVerbose,
												OptimizeForSize,
												debugNativeOutput,
												allWarnings,
												functionSectioning,
												dataSectioning,
												onlyHeaderFileOutput,
												noStandardCLibraries,
												microcontrollerUnitSpec<>arduinoChipNumber,
												gnuC11Std
											},
											" "]
										],
									"Defines"->
									Join[
										{
											"SERIAL_RX_BUFFER_SIZE="<>"128",
											"SERIAL_TX_BUFFER_SIZE="<>"64",
											clockCrystalSpeedDef<>clockSpeed,
											arduinoBoard,
											arduinoVersionDef<>arduinoVersion
										},
										extraCompileDefines
									],
									"LibraryDirectories" -> Join[arduinoStandardLibraryDirectories,{FileNameJoin[{buildLocation,"libs"}]}],
									"IncludeDirectories" -> Join[{avrGCCIncludeLocation,PacletResource["DeviceDriver_Arduino","avr-libc-include"],FileNameJoin[{buildLocation,"libs"}]},arduinoStandardLibraryDirectories],
									"ShellOutputFunction"->(compilationProgressCatcher[#,"output"]&),
									"ShellCommandFunction"->(compilationProgressCatcher[#,"command","Debug"->OptionValue["Debug"]]&),
									"CleanIntermediate"->False,
									(*SystemIncludeDirectories, SystemLibraryDirectories, and SystemLibraries are for
						  			Wolfram C Libraries, so they can be discarded*)
						  			"SystemIncludeDirectories" -> {}
								];
							),
							ToLowerCase[FileExtension[lib]]==="cpp",
							(*the library is a C++ library, so compile it with g++*)
							(
								If[OptionValue["Debug"],
									Print[lib," is a c++ library"];
								];
								CreateObjectFile[
									{
										FileNameJoin[{buildLocation,"libs",lib}]
									},
									lib,
									"Compiler"->AVRCCompiler,
									"CompilerInstallation"->compilerLocation,
									"CompilerName"->"avr-g++",
									"TargetDirectory"->buildLocation,
									"CompileOptions"->CommandJoin[
										Riffle[
											{
												highVerbose,
												OptimizeForSize,
												debugNativeOutput,
												allWarnings,
												noExceptions,
												functionSectioning,
												dataSectioning,
												noThreading,
												onlyHeaderFileOutput,
												noStandardCLibraries,
												microcontrollerUnitSpec<>arduinoChipNumber,
												gnuCPP11Std
											},
											" "]
										],
									"Defines"->
									Join[
										{
											"SERIAL_RX_BUFFER_SIZE="<>"128",
											"SERIAL_TX_BUFFER_SIZE="<>"64",
											clockCrystalSpeedDef<>clockSpeed,
											arduinoBoard,
											arduinoVersionDef<>arduinoVersion
										},
										extraCompileDefines
									],
									"LibraryDirectories" -> Join[arduinoStandardLibraryDirectories,{FileNameJoin[{buildLocation,"libs"}]}],
									"IncludeDirectories" -> Join[{avrGCCIncludeLocation,PacletResource["DeviceDriver_Arduino","avr-libc-include"],FileNameJoin[{buildLocation,"libs"}]},arduinoStandardLibraryDirectories],
									"ShellOutputFunction"->(compilationProgressCatcher[#,"output"]&),
									"ShellCommandFunction"->(compilationProgressCatcher[#,"command","Debug"->OptionValue["Debug"]]&),
									"CleanIntermediate"->False,
									(*SystemIncludeDirectories, SystemLibraryDirectories, and SystemLibraries are for
						  			Wolfram C Libraries, so they can be discarded*)
						  			"SystemIncludeDirectories" -> {}
								]
							),
							ToLowerCase[FileExtension[lib]]==="s",
							(*library is an assembly file*)
							(
								CreateObjectFile[
									{
										FileNameJoin[{buildLocation,"libs",lib}]
									},
									lib,
									"Compiler"->AVRCCompiler,
									"CompilerInstallation"->compilerLocation,
									"CompilerName"->"avr-gcc",
									"TargetDirectory"->buildLocation,
									"CompileOptions"->CommandJoin[
										Riffle[
											{
												languageSpec<>" "<>assembly,
												highVerbose,
												OptimizeForSize,
												debugNativeOutput,
												allWarnings,
												microcontrollerUnitSpec<>arduinoChipNumber,
												noStandardCLibraries
											},
											" "]
										],
									"Defines"->
										{
											"SERIAL_RX_BUFFER_SIZE="<>"128",
											"SERIAL_TX_BUFFER_SIZE="<>"64",
											clockCrystalSpeedDef<>clockSpeed,
											arduinoVersionDef<>arduinoVersion,
											arduinoBoard
										},
									"LibraryDirectories" -> Join[arduinoStandardLibraryDirectories,{FileNameJoin[{buildLocation,"libs"}]}],
						  			"IncludeDirectories" -> Join[{avrGCCIncludeLocation,PacletResource["DeviceDriver_Arduino","avr-libc-include"],FileNameJoin[{buildLocation,"libs"}]},arduinoStandardLibraryDirectories],
							  		"ShellOutputFunction"->(compilationProgressCatcher[#,"output"]&),
									"ShellCommandFunction"->(compilationProgressCatcher[#,"command","Debug"->False]&),
									"CleanIntermediate"->False,
									(*SystemIncludeDirectories, SystemLibraryDirectories, and SystemLibraries are for
								  	Wolfram C Libraries, so they can be discarded*)
						  			"SystemIncludeDirectories" -> {}
								]
							)
						];
						(*after compiling the library, check the file to see if it exists and if it does move it*)
						If[fileCheck[getWolfWorkDir[buildLocation],lib<>".o"],
							(*THEN*)
							(*the file was compiled and the output exists*)
							moveLibraryFiles[getWolfWorkDir[buildLocation],lib],
							(*ELSE*)
							(*the file doesn't exist, so the compilation failed*)
							(*Raise Message and throw $Failed*)
							(
								Message[arduinoCompile::cLibraryFail,lib];
								If[$OperatingSystem === "Windows",
									SetEnvironment["PATH"->originalPathEnvironment];
								];
								Return[$Failed];
							)
						];
						
					),
					{lib,libs}
				];
				
				(*now compile the standard libraries*)				
				$arduinoStandardLibraryFiles=compileArduinoStandardLibraries[
					arduinoInstallLocation,
					buildLocation,
					"AVRGCCLocation"->OptionValue["AVRGCCLocation"],
					"StandardArduinoLibrariesLocation"->OptionValue["StandardArduinoLibrariesLocation"],
					"ArduinoVersion"->arduinoVersion,
					"ChipPartNumber"->arduinoChipNumber,
					"ArduinoBoardDefine"->arduinoBoard,
					"ExtraCompileDefines"->extraCompileDefines
				];
				
				(*check the resultof the standard libraries*)
				If[$arduinoStandardLibraryFiles===$Failed,
					(*THEN*)
					(*the libraries failed, but a message should already have been generated, so just reset the path environment variable and exit*)
					(
						If[$OperatingSystem === "Windows",
							SetEnvironment["PATH"->originalPathEnvironment];
						];
						Return[$Failed];
					)
				];
				
				(*increment the progress indiciator for the progress bar*)
				ArduinoLink`Private`$compilationProgressIndicator++;
				
				(*finally compile all of the files into the .elf file*)
				
				CreateExecutable[
					Join[{FileNameJoin[{buildLocation,fileName<>".o"}]},FileNameJoin[{buildLocation,"liboutput",#<>".o"}]&/@Reverse[libs]],
					fileName,
					"Compiler"->AVRCCompiler,
					"CompilerInstallation"->compilerLocation,
					"CompilerName"->"avr-gcc",
					"TargetDirectory"->buildLocation,
					"CompileOptions" ->CommandJoin[
						Riffle[
							{
								lowVerbose,
								OptimizeForSize,
								allWarnings,
								noStandardCLibraries,
								noStandartStartFiles,
								microcontrollerUnitSpec<>arduinoChipNumber,
								"-Wl,"<>linkerDirectiveOnlyUsedCode<>","<>linkerDirectiveVerboseOutput,
								gnuC11Std
							},
							" "]
						],
					"ShellOutputFunction"->(compilationProgressCatcher[#,"output"]&),
					"ShellCommandFunction"->(compilationProgressCatcher[#,"command","Debug"->OptionValue["Debug"]]&),
					"CleanIntermediate" -> False,
					"LibraryDirectories"->{
						FileNameJoin[{PacletResource["DeviceDriver_Arduino","avr-libc-lib"],avrChipPartNumberToArchitecture[arduinoChipNumber]}],
						FileNameJoin[{buildLocation,"liboutput"}],
						buildLocation
					},
					(*SystemIncludeDirectories, SystemLibraryDirectories, and SystemLibraries are for
				  	Wolfram C Libraries, so they can be discarded*)
					"SystemIncludeDirectories" -> {},
					"SystemLibraryDirectories" -> {},
					"SystemLibraries" -> {},
					"ExtraObjectFiles"->Join[
						{
							FileNameJoin[
								{
									PacletResource["DeviceDriver_Arduino","avr-libc-lib"],
									avrChipPartNumberToArchitecture[arduinoChipNumber],
									avrChipNumberToCRTFile[arduinoChipNumber]
								}
							]
						},
						$arduinoStandardLibraryFiles
					],
					(*m is the math library for avr platform*)
					"Libraries"->"m"
				];
				
				(*increment the progress indiciator for the progress bar*)
				ArduinoLink`Private`$compilationProgressIndicator++;
				
				If[fileCheck[getWolfWorkDir[buildLocation],fileName<>".elf"],
					(*THEN*)
					(*the file was compiled and the output exists*)
					moveFiles[getWolfWorkDir[buildLocation]],
					(*ELSE*)
					(*the file doesn't exist, so the compilation failed*)
					(*Raise Message and throw $Failed*)
					(
						Message[arduinoCompile::execFailed];
						If[$OperatingSystem === "Windows",
							SetEnvironment["PATH"->originalPathEnvironment];
						];
						Return[$Failed];
					)
				]
			)		
		];
		
		If[OptionValue["Debug"]===True,
			(
				Print["Finished compiling..."];
				Print["Took ",AbsoluteTime[] - $startCompileTime," seconds to compile"];
			)
		];
		
		(*finally reset the environment path before exiting*)
		If[$OperatingSystem === "Windows",
			SetEnvironment["PATH"->originalPathEnvironment];
		];

		Return[$output]
	)
];

Options[compileArduinoStandardLibraries]=
	{
		"LibraryCoreLocation"->Default,
		"AVRGCCLocation"->Default,
		"StandardArduinoLibrariesLocation"->Default,
		"ArduinoVersion"->"10603",
		"ChipPartNumber"->"atmega328p",
		"ArduinoBoardDefine"->"ARDUINO_AVR_UNO",
		"ExtraCompileDefines"->{}
	};
(*this function will compile all of the standard arduino libraries and return a list of the filenames of the object files from all of the libraries*)
compileArduinoStandardLibraries[arduinoInstallLocation_,buildLocation_,OptionsPattern[]]:=Module[
	{
		highVerbose= CommandJoin[Table["-v ",{4}]],
		OptimizeForSize = "-Os",
		debugNativeOutput="-g",
		allWarnings = "-Wall",
		noExceptions = "-fno-exceptions",
		functionSectioning = "-ffunction-sections",
		dataSectioning = "-fdata-sections",
		noThreading="-fno-threadsafe-statics",
		microcontrollerUnitSpec = "-mmcu=",
		arduinoChipNumber = OptionValue["ChipPartNumber"],
		clockCrystalSpeedDef = "F_CPU=",
		clockSpeed ="16000000L",
		onlyHeaderFileOutput ="-MMD",
		arduinoVersionDef = "ARDUINO=",
		arduinoVersion = OptionValue["ArduinoVersion"],
		arduinoBoard=OptionValue["ArduinoBoardDefine"],
		libraryCoreLocation,
		compilerLocation,
		arduinoStandardLibraryDirectories,
		noStandardCLibraries="-nostdinc",
		avrGCCIncludeLocation,
		languageSpec="-x",
		assembly="assembler-with-cpp",
		extraCompileDefines=OptionValue["ExtraCompileDefines"],
		gnuCPP11Std="-std=gnu++11",
		gnuC11Std="-std=gnu11"
	},
	(
		compilerLocation = OptionValue["AVRGCCLocation"];
		
		arduinoStandardLibraryDirectories = OptionValue["StandardArduinoLibrariesLocation"];
		
		avrGCCIncludeLocation=FileNameJoin[
			{
				FileNameDrop[compilerLocation],
				"lib",
				"gcc",
				"avr",
				(*this figures out the version number of gcc*)
				FileNameTake[First[FileNames["*",FileNameJoin[{FileNameDrop[compilerLocation],"lib","gcc","avr"}]]]],
				"include"
			}
		];
	
		(*the files we need to compile are located in the first directory with all the standard libraries*)
		libraryCoreLocation = First[Select[FixedPoint[FileBaseName,#]=="arduino"&]@arduinoStandardLibraryDirectories];
	
		compiledLibs={};
		(*dynamically get the names of the files to compile, this is generated from the libraryCoreLocation*)
		arduinoLibraries = FileNameTake[#,-1]&/@Union[
			FileNames["*.c", libraryCoreLocation],
			FileNames["*.cpp",libraryCoreLocation],
			FileNames["*.s",libraryCoreLocation]
		];
		(*compile all the files necessary*)
		(*two different cases, one with gcc, the other with g++*)
		For[libNum = 1, libNum <= Length[arduinoLibraries], libNum++,
			(
				(*increment the progress indiciator for the progress bar*)
				ArduinoLink`Private`$compilationProgressIndicator++;
				lib = arduinoLibraries[[libNum]];
				Which[ToUpperCase[FileExtension[lib]]==="C",
					(*it is a c library, so compile it with gcc*)
					(
						CreateObjectFile[
							{
								FileNameJoin[{libraryCoreLocation,lib}]
							},
							lib,
							"Compiler"->AVRCCompiler,
							"CompilerInstallation"->compilerLocation,
							"CompilerName"->"avr-gcc",
							"TargetDirectory"->buildLocation,
							"CompileOptions"->CommandJoin[
								Riffle[
									{
										highVerbose,
										OptimizeForSize,
										debugNativeOutput,
										allWarnings,
										functionSectioning,
										dataSectioning,
										onlyHeaderFileOutput,
										noStandardCLibraries,
										microcontrollerUnitSpec<>arduinoChipNumber,
										gnuC11Std
									},
									" "]
								],
							"Defines"->Join[
								{
									"SERIAL_RX_BUFFER_SIZE="<>"128",
									"SERIAL_TX_BUFFER_SIZE="<>"64",
									clockCrystalSpeedDef<>clockSpeed,
									arduinoBoard,
									arduinoVersionDef<>arduinoVersion
								},
								extraCompileDefines
							],
							"LibraryDirectories" -> arduinoStandardLibraryDirectories,
				  			"IncludeDirectories" -> Join[{avrGCCIncludeLocation,PacletResource["DeviceDriver_Arduino","avr-libc-include"]},arduinoStandardLibraryDirectories],
					  		"ShellOutputFunction"->(compilationProgressCatcher[#,"output"]&),
							"ShellCommandFunction"->(compilationProgressCatcher[#,"command","Debug"->False]&),
							"CleanIntermediate"->False,
							(*SystemIncludeDirectories, SystemLibraryDirectories, and SystemLibraries are for
						  	Wolfram C Libraries, so they can be discarded*)
				  			"SystemIncludeDirectories" -> {}
						]
					),
					ToUpperCase[FileExtension[lib]]==="CPP",
					(*it is a c++ library, so compile it with g++*)
					(
						CreateObjectFile[
							{
								FileNameJoin[{libraryCoreLocation,lib}]
							},
							lib,
							"Compiler"->AVRCCompiler,
							"CompilerInstallation"->compilerLocation,
							"CompilerName"->"avr-g++",
							"TargetDirectory"->buildLocation,
							"CompileOptions"->CommandJoin[
								Riffle[
									{
										highVerbose,
										OptimizeForSize,
										debugNativeOutput,
										allWarnings,
										noExceptions,
										functionSectioning,
										dataSectioning,
										noThreading,
										onlyHeaderFileOutput,
										noStandardCLibraries,
										microcontrollerUnitSpec<>arduinoChipNumber,
										gnuCPP11Std
									},
									" "]
								],
							"Defines"->Join[
								{
									"SERIAL_RX_BUFFER_SIZE="<>"128",
									"SERIAL_TX_BUFFER_SIZE="<>"64",
									clockCrystalSpeedDef<>clockSpeed,
									arduinoBoard,
									arduinoVersionDef<>arduinoVersion
								},
								extraCompileDefines
							],
							"LibraryDirectories" -> arduinoStandardLibraryDirectories,
				  			"IncludeDirectories" -> Join[{avrGCCIncludeLocation,PacletResource["DeviceDriver_Arduino","avr-libc-include"]},arduinoStandardLibraryDirectories],
				  			"ShellOutputFunction"->(compilationProgressCatcher[#,"output"]&),
							"ShellCommandFunction"->(compilationProgressCatcher[#,"command","Debug"->False]&),
							"CleanIntermediate"->False,
							(*SystemIncludeDirectories, SystemLibraryDirectories, and SystemLibraries are for
						  	Wolfram C Libraries, so they can be discarded*)
				  			"SystemIncludeDirectories" -> {}
						]
					),
					ToUpperCase[FileExtension[lib]]==="S",
					(*it is an assembly file*)
					(
						CreateObjectFile[
							{
								FileNameJoin[{libraryCoreLocation,lib}]
							},
							lib,
							"Compiler"->AVRCCompiler,
							"CompilerInstallation"->compilerLocation,
							"CompilerName"->"avr-gcc",
							"TargetDirectory"->buildLocation,
							"CompileOptions"->CommandJoin[
								Riffle[
									{
										languageSpec<>" "<>assembly,
										highVerbose,
										OptimizeForSize,
										debugNativeOutput,
										allWarnings,
										microcontrollerUnitSpec<>arduinoChipNumber,
										noStandardCLibraries
									},
									" "]
								],
							"Defines"->
								{
									"SERIAL_RX_BUFFER_SIZE="<>"128",
									"SERIAL_TX_BUFFER_SIZE="<>"64",
									clockCrystalSpeedDef<>clockSpeed,
									arduinoVersionDef<>arduinoVersion,
									arduinoBoard
								},
							"LibraryDirectories" -> arduinoStandardLibraryDirectories,
				  			"IncludeDirectories" -> Join[{avrGCCIncludeLocation,PacletResource["DeviceDriver_Arduino","avr-libc-include"]},arduinoStandardLibraryDirectories],
					  		"ShellOutputFunction"->(compilationProgressCatcher[#,"output"]&),
							"ShellCommandFunction"->(compilationProgressCatcher[#,"command","Debug"->False]&),
							"CleanIntermediate"->False,
							(*SystemIncludeDirectories, SystemLibraryDirectories, and SystemLibraries are for
						  	Wolfram C Libraries, so they can be discarded*)
				  			"SystemIncludeDirectories" -> {}
						]
					)
				];
				(*THE FOLLOWING IS A WORKAROUND FOR A WIERD BUG WHERE CCOMPILERDRIVER THINKS IT FAILED EVEN WHEN 
				IT SUCCESSFULLY COMPLETES SO HERE WE MANUALLY CHECK TO SEE IF IT FAILED*)
				If[fileCheck[getWolfWorkDir[buildLocation],lib<>".o"],
					(*THEN*)
					(*the file was compiled and the output exists*)
					(
						moveFiles[getWolfWorkDir[buildLocation]];
						AppendTo[compiledLibs,FileNameJoin[{buildLocation,lib<>".o"}]]
					),
					(*ELSE*)
					(*the file doesn't exist, so the compilation failed*)
					(*Raise Message and throw $Failed*)
					(
						Message[arduinoCompile::arduinoLibraryFail,lib];
						Return[$Failed];
					)
				];
			)
		];
		Return[compiledLibs];
	)
]



(*cLibFinder will take as an argument the folder to search, and will return a list of all of the files with .cpp or .h extensions*)
libFinder[libFolder_]:=Module[{},
	(
		files=FileNames[{"*.c","*.cpp","*.s"},FileNameJoin[{libFolder,"libs"}]];
		Return[Last[FileNameSplit[#]]&/@files]
	)
];


fileCheck[locationToCheck_,fileToCheckFor_]:=Module[
	{
		dirFiles = (Last@FileNameSplit[#])&/@FileNames["*",locationToCheck]
	},
	(
		MemberQ[dirFiles,fileToCheckFor]
	)
]



(*go and get the actual folder name from CCompilerDriverBase for this*)
getWolfWorkDir[location_]:=FileNameJoin[{location,
	StringJoin["Working-", 
		$MachineName, "-", 
		ToString[$ProcessID], "-",
		ToString[Developer`ThreadID[]], "-",
		ToString[CCompilerDriver`CCompilerDriverBase`Private`$WorkingDirCount]]}];



(*first copy all of the files to the normal working directory, then delete the directory passed*)
moveFiles[wolfFolder_]:=Module[{buildLocation = FileNameJoin@Most@FileNameSplit@wolfFolder},
	(
		Quiet[CopyFile[#,FileNameJoin[{buildLocation,Last@FileNameSplit@#}]] & /@
				(Select[(FileExtension[#] != "bat")&]@FileNames["*", wolfFolder])];
		DeleteDirectory[wolfFolder,"DeleteContents"->True];
	)
];


(*does just about the same thing as the regular move Files, but this will be smart about handling the 
libraries by copying the files into the correct library*)
moveLibraryFiles[wolfFolder_,libName_]:=Module[{buildLocation = FileNameJoin@Most@FileNameSplit@wolfFolder},
	(
		Quiet[CopyFile[#, FileNameJoin[{buildLocation,"liboutput",Last@FileNameSplit@#}]] & /@
				(Select[(FileExtension[#] != "bat")&]@FileNames["*", wolfFolder])];
		DeleteDirectory[wolfFolder,"DeleteContents"->True];
	)
];



(*set $output to be empty initially, then add to it rules of the form input -> output as they arrive*)
$output= <||>;
$prevCommand = "";
Options[compilationProgressCatcher]={"Debug"->False}
compilationProgressCatcher[string_,type_,OptionsPattern[]]:=Module[{},
	(
		If[ToLowerCase[type]==="command",
			(*THEN*)
			(*the string is a command, so set the $prevCommand to it, and if debugging is on, print it off*)
			(
				$prevCommand = string;
				If[TrueQ[OptionValue["Debug"]],
					(*THEN*)
					(*print off the command*)
					Print["Compiling with command :\n",string];
				]
			),
			(*ELSE*)
			(*the string is the output of the previous command stored in $prevCommand, so add the pair to the output*)
			(*check to make sure it is output though*)
			(
				If[ToLowerCase[type]==="output",
					(*THEN*)
					(*it is good, add it to the $output*)
					AppendTo[$output,$prevCommand->string];
					(*ELSE*)
					(*don't do anything*)
				]
			)
		]
	)
];

(*unfortunatley there's not a nice table for this, and it's not obviously just a formula that converts the chip number into what gcc calls the object file, so this will have to be*)
(*manually populated*)
avrChipNumberToCRTFile=
<|
	"atmega328p"->"crtm328p.o","atmega32u4"->"crtm32u4.o"
|>;

(*this is from the online documentation for avr-libc that has a table of this information, which is accurate through avr-gcc 4.3 (there might be more specific part numbers for later 
gcc versions, but these will still work)*)
avrChipPartNumberToArchitecture=<|"at90s1200" -> "avr1", "attiny11" -> "avr1", "attiny12" -> "avr1", 
 "attiny15" -> "avr1", "attiny28" -> "avr1", "at90s2313" -> "avr2", 
 "at90s2323" -> "avr2", "at90s2333" -> "avr2", "at90s2343" -> "avr2", 
 "attiny22" -> "avr2", "attiny26" -> "avr2", "at90s4414" -> "avr2", 
 "at90s4433" -> "avr2", "at90s4434" -> "avr2", "at90s8515" -> "avr2", 
 "at90c8534" -> "avr2", "at90s8535" -> "avr2", "at86rf401" -> "avr25",
  "ata5272" -> "avr25", "attiny13" -> "avr25", "attiny13a" -> "avr25",
  "attiny2313" -> "avr25", "attiny2313a" -> "avr25", 
 "attiny24" -> "avr25", "attiny24a" -> "avr25", "attiny25" -> "avr25",
  "attiny261" -> "avr25", "attiny261a" -> "avr25", 
 "attiny4313" -> "avr25", "attiny43u" -> "avr25", 
 "attiny44" -> "avr25", "attiny44a" -> "avr25", "attiny45" -> "avr25",
  "attiny461" -> "avr25", "attiny461a" -> "avr25", 
 "attiny48" -> "avr25", "attiny828" -> "avr25", "attiny84" -> "avr25",
  "attiny84a" -> "avr25", "attiny85" -> "avr25", 
 "attiny861" -> "avr25", "attiny861a" -> "avr25", 
 "attiny87" -> "avr25", "attiny88" -> "avr25", "atmega603" -> "avr3", 
 "at43usb355" -> "avr3", "atmega103" -> "avr31", 
 "at43usb320" -> "avr31", "at90usb82" -> "avr35", 
 "at90usb162" -> "avr35", "ata5505" -> "avr35", 
 "atmega8u2" -> "avr35", "atmega16u2" -> "avr35", 
 "atmega32u2" -> "avr35", "attiny167" -> "avr35", 
 "attiny1634" -> "avr35", "at76c711" -> "avr3", "ata6285" -> "avr4", 
 "ata6286" -> "avr4", "ata6289" -> "avr4", "atmega48" -> "avr4", 
 "atmega48a" -> "avr4", "atmega48pa" -> "avr4", "atmega48p" -> "avr4",
  "atmega8" -> "avr4", "atmega8a" -> "avr4", "atmega8515" -> "avr4", 
 "atmega8535" -> "avr4", "atmega88" -> "avr4", "atmega88a" -> "avr4", 
 "atmega88p" -> "avr4", "atmega88pa" -> "avr4", 
 "atmega8hva" -> "avr4", "at90pwm1" -> "avr4", "at90pwm2" -> "avr4", 
 "at90pwm2b" -> "avr4", "at90pwm3" -> "avr4", "at90pwm3b" -> "avr4", 
 "at90pwm81" -> "avr4", "at90can32" -> "avr5", "at90can64" -> "avr5", 
 "at90pwm161" -> "avr5", "at90pwm216" -> "avr5", 
 "at90pwm316" -> "avr5", "at90scr100" -> "avr5", 
 "at90usb646" -> "avr5", "at90usb647" -> "avr5", "at94k" -> "avr5", 
 "atmega16" -> "avr5", "ata5790" -> "avr5", "ata5795" -> "avr5", 
 "atmega161" -> "avr5", "atmega162" -> "avr5", "atmega163" -> "avr5", 
 "atmega164a" -> "avr5", "atmega164p" -> "avr5", 
 "atmega164pa" -> "avr5", "atmega165" -> "avr5", 
 "atmega165a" -> "avr5", "atmega165p" -> "avr5", 
 "atmega165pa" -> "avr5", "atmega168" -> "avr5", 
 "atmega168a" -> "avr5", "atmega168p" -> "avr5", 
 "atmega168pa" -> "avr5", "atmega169" -> "avr5", 
 "atmega169a" -> "avr5", "atmega169p" -> "avr5", 
 "atmega169pa" -> "avr5", "atmega16a" -> "avr5", 
 "atmega16hva" -> "avr5", "atmega16hva2" -> "avr5", 
 "atmega16hvb" -> "avr5", "atmega16hvbrevb" -> "avr5", 
 "atmega16m1" -> "avr5", "atmega16u4" -> "avr5", "atmega32" -> "avr5",
  "atmega32a" -> "avr5", "atmega323" -> "avr5", 
 "atmega324a" -> "avr5", "atmega324p" -> "avr5", 
 "atmega324pa" -> "avr5", "atmega325" -> "avr5", 
 "atmega325a" -> "avr5", "atmega325p" -> "avr5", 
 "atmega325pa" -> "avr5", "atmega3250" -> "avr5", 
 "atmega3250a" -> "avr5", "atmega3250p" -> "avr5", 
 "atmega3250pa" -> "avr5", "atmega328" -> "avr5", 
 "atmega328p" -> "avr5", "atmega329" -> "avr5", 
 "atmega329a" -> "avr5", "atmega329p" -> "avr5", 
 "atmega329pa" -> "avr5", "atmega3290" -> "avr5", 
 "atmega3290a" -> "avr5", "atmega3290p" -> "avr5", 
 "atmega3290pa" -> "avr5", "atmega32c1" -> "avr5", 
 "atmega32hvb" -> "avr5", "atmega32hvbrevb" -> "avr5", 
 "atmega32m1" -> "avr5", "atmega32u4" -> "avr5", 
 "atmega32u6" -> "avr5", "atmega406" -> "avr5", 
 "atmega644rfr2" -> "avr5", "atmega64rfr2" -> "avr5", 
 "atmega64" -> "avr5", "atmega64a" -> "avr5", "atmega640" -> "avr5", 
 "atmega644" -> "avr5", "atmega644a" -> "avr5", 
 "atmega644p" -> "avr5", "atmega644pa" -> "avr5", 
 "atmega645" -> "avr5", "atmega645a" -> "avr5", 
 "atmega645p" -> "avr5", "atmega6450" -> "avr5", 
 "atmega6450a" -> "avr5", "atmega6450p" -> "avr5", 
 "atmega649" -> "avr5", "atmega649a" -> "avr5", 
 "atmega6490" -> "avr5", "atmega6490a" -> "avr5", 
 "atmega6490p" -> "avr5", "atmega649p" -> "avr5", 
 "atmega64c1" -> "avr5", "atmega64hve" -> "avr5", 
 "atmega64m1" -> "avr5", "m3000" -> "avr5", "at90can128" -> "avr51", 
 "at90usb1286" -> "avr51", "at90usb1287" -> "avr51", 
 "atmega128" -> "avr51", "atmega128a" -> "avr51", 
 "atmega1280" -> "avr51", "atmega1281" -> "avr51", 
 "atmega1284" -> "avr51", "atmega1284p" -> "avr51", 
 "atmega1284rfr2" -> "avr51", "atmega128rfr2" -> "avr51", 
 "atmega2560" -> "avr6", "atmega2561" -> "avr6", 
 "atmega2564rfr2" -> "avr6", "atmega256rfr2" -> "avr6", 
 "atxmega16a4" -> "avrxmega2", "atxmega16a4u" -> "avrxmega2", 
 "atxmega16c4" -> "avrxmega2", "atxmega16d4" -> "avrxmega2", 
 "atxmega32a4" -> "avrxmega2", "atxmega32a4u" -> "avrxmega2", 
 "atxmega32c4" -> "avrxmega2", "atxmega32d4" -> "avrxmega2", 
 "atxmega64a3" -> "avrxmega4", "atxmega64a3u" -> "avrxmega4", 
 "atxmega64a4u" -> "avrxmega4", "atxmega64b1" -> "avrxmega4", 
 "atxmega64b3" -> "avrxmega4", "atxmega64c3" -> "avrxmega4", 
 "atxmega64d3" -> "avrxmega4", "atxmega64d4" -> "avrxmega4", 
 "atxmega64a1" -> "avrxmega5", "atxmega64a1u" -> "avrxmega5", 
 "atxmega128a3" -> "avrxmega6", "atxmega128a3u" -> "avrxmega6", 
 "atxmega128b1" -> "avrxmega6", "atxmega128b3" -> "avrxmega6", 
 "atxmega128c3" -> "avrxmega6", "atxmega128d3" -> "avrxmega6", 
 "atxmega128d4" -> "avrxmega6", "atxmega192a3" -> "avrxmega6", 
 "atxmega192a3u" -> "avrxmega6", "atxmega192c3" -> "avrxmega6", 
 "atxmega192d3" -> "avrxmega6", "atxmega256a3" -> "avrxmega6", 
 "atxmega256a3u" -> "avrxmega6", "atxmega256a3b" -> "avrxmega6", 
 "atxmega256a3bu" -> "avrxmega6", "atxmega256c3" -> "avrxmega6", 
 "atxmega256d3" -> "avrxmega6", "atxmega384c3" -> "avrxmega6", 
 "atxmega384d3" -> "avrxmega6", "atxmega128a1" -> "avrxmega7", 
 "atxmega128a1u" -> "avrxmega7", "atxmega128a4u" -> "avrxmega7", 
 "attiny4" -> "avrtiny10", "attiny5" -> "avrtiny10", 
 "attiny9" -> "avrtiny10", "attiny10" -> "avrtiny10", 
 "attiny20" -> "avrtiny10", "attiny40" -> "avrtiny10"|>



End[] (* End Private Context *)

EndPackage[]
