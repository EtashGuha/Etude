(* ::Package:: *)

(*:Mathematica Version: 2.2 *)

(*:Name: Audio` *)

(*:Title: Audio Functions *)

(*:Author: Arun Chandra (Wolfram Research), September 1992.  *)

(*:Summary:
This package provides functions for the manipulation and synthesis of sounds.
*)

(*:Context: Audio` *)

(* :Copyright: Copyright 1992-2007,  Wolfram Research, Inc.
*)

(*:Package Version: 1.2 *)

(* :History:  original version by Arun Chandra,
		V1.1, changed Type to ModulationType and
		Ring to RingModulation to avoid symbol conflicts.
		V1.2 June 1997 by John M. Novak, corrected major problems in Wave
		    and AIFF reading, with input from Terry Robb and others.
        September 2006 by Brian Van Vertloo, updated to fit the new
            paclet format for Mathematica 6.0, and updated since
            Utilities`FilterOptions` is obsolete.
*)

(*:Reference: Usage messages only. *)

(*:Keywords: sound, music, synthesis, composition *)

(*:Requirements: A system on which Mathematica can produce sound. *)

(*:Warning: The MS RIFF reader currently searches for a single format
   chunk; then reads the first data chunk that follows it. It does not
   handle multiple data chunks, or other information within the RIFF
   format. It only handles PCM coded waveforms.
 *)

(*:Sources: 
    Brun, Herbert. 1991. My Words and Where I Want Them. London:
        Princelet Editions.
    Dodge, Charles. 1985. Computer Music.  New York: Schirmer Books.
    Hiller, Lejaren A. 1963-66. Lectures on Musical Acoustics. Unpublished.
    Mathews, Max V. 1969. The Technology of Computer Music. 
        Cambridge, MA: MIT Press.
    Moore, F. Richard. 1990. Elements of Computer Music. 
        Englewood Cliffs, NJ: Prentice-Hall.
    Olson, Harry F. 1967. Music, Physics, and Engineering. 
        New York: Dover Publications, Inc.
    Wells, Thomas H. 1981. The Technology of Electronic Music. 
        New York: Schirmer Books.
*)
(* Attach following to General for convenience; easier for users to deactivate. *)
General::compat = "Audio functionality has been superseded by preloaded functionality. The package now being loaded may conflict with this. Please see the Compatibility Guide for details.";
Message[General::compat];

BeginPackage["Audio`"];

Unprotect[ Waveform, AmplitudeModulation, FrequencyModulation, ReadSoundFile ];

If[Not@ValueQ[Waveform::usage],Waveform::usage = 
"Waveform[t, f, d, opts] returns a Sound object of waveform type \
t, having a fundamental frequency f Hertz, and duration of d seconds. \
The type t must be one of the following: Sinusoid, Sawtooth, Square, \
Triangle. The option Overtones sets the number of overtones that \
will be present in the sound."];

If[Not@ValueQ[ListWaveform::usage],ListWaveform::usage = 
"ListWaveform[{{n1,a1},{n2,a2},...}, f, d, opts] returns a Sound \
object having a fundamental frequency f Hertz and duration of d \
seconds, where ni is a frequency relative to the fundamental and \
ai is the relative amplitude of that frequency."];

If[Not@ValueQ[Overtones::usage],Overtones::usage = "Overtones is an option to Waveform that specifies the  \
number of overtones to be present in a standard waveform when created with  \
Fourier summation."];

If[Not@ValueQ[Sinusoid::usage],Sinusoid::usage = 
"Sinusoid is a type of waveform."];
If[Not@ValueQ[Sawtooth::usage],Sawtooth::usage = 
"Sawtooth is a type of waveform."];
If[Not@ValueQ[Square::usage],Square::usage = 
"Square is a type of waveform."];
If[Not@ValueQ[Triangle::usage],Triangle::usage = 
"Triangle is a type of waveform."];

If[Not@ValueQ[AmplitudeModulation::usage],AmplitudeModulation::usage = 
"AmplitudeModulation[fc, fm, mi, d, opts] returns a Sound object \
that is an amplitude modulated sinusoid, with fc and fm being the \
carrier and modulating frequencies in Hertz, mi the modulation \
index, and d the duration of the sound in seconds. If the option  \
RingModulation is set to True, the sound will be ring-modulated."];

If[Not@ValueQ[Ring::usage],Ring::usage =
"Ring is an an obsolete option to AmplitudeModulation, replaced by \
RingModulation."];

If[Not@ValueQ[RingModulation::usage],RingModulation::usage =
"RingModulation is an option to AmplitudeModulation. When RingModulation -> \
True, the Sound object will contain only the sum and difference of the \
carrier and modulating frequencies."];

If[Not@ValueQ[FrequencyModulation::usage],FrequencyModulation::usage = 
"FrequencyModulation[fc, {fm, pd}, d, opts] returns a Sound object \
that is a frequency modulated sinusoid, where fc, fm, and pd are \
the carrier, modulator, and peak deviation frequencies in Hertz, \
and d is the duration in seconds. The option ModulationType can be set \
to Standard (default), Cascade, or Parallel. For Cascade \
and Parallel, the second argument must be a list of pairs of the form \
{{fm1, pd1}, {fm2, pd2}, ...}, where pdi is the peak deviation associated \
with modulating frequency fmi, and both values are measured in Hertz."];

If[Not@ValueQ[ModulationType::usage],ModulationType::usage =
"ModulationType is an option of FrequencyModulation, specifying the \
type of modulation: Standard, Cascade, or Parallel. Standard frequency \
modulation is specified by two parameters {fm, pd} giving the \
modulating frequency and the peak deviation. Cascade and Parallel \
modulation is specified by a list of parameters {{f1, pd1}, {f2, pd2}, ...} \
giving the modulating frequencies and peak deviations of the \
cascaded or parallel modulations."];

If[Not@ValueQ[Standard::usage],Standard::usage =
"Standard is a possible value of ModulationType, an option of \
FrequencyModulation. Standard frequency modulation is described by \
two parameters: the modulating frequency and the peak deviation."]; 

If[Not@ValueQ[Cascade::usage],Cascade::usage =
"Cascade is a possible value of ModulationType, an option of \
FrequencyModulation. In cascade modulation, the modulating frequency is \
itself modulated by second frequency, which may be modulated by a third \
frequency, and so on."];

If[Not@ValueQ[Parallel::usage],Parallel::usage =
"Parallel is a possible value of ModulationType, an option of \
FrequencyModulation. In parallel modulation, the carrier frequency is \
modulated by two or more frequencies in parallel."];

If[Not@ValueQ[ReadSoundFile::usage],ReadSoundFile::usage =
"ReadSoundFile[\"soundfile\"] reads the specified sound file, and returns a \
list of amplitudes between -32768 and +32767. If the option \
PrintHeader is set to True, the header information in the \
sound file (the sampling rate, sample width, etc.) will be displayed."];

If[Not@ValueQ[ReadSoundfile::usage],ReadSoundfile::usage = 
"Obsolete name. Use ReadSoundFile instead."];

If[Not@ValueQ[PrintHeader::usage],PrintHeader::usage = 
"PrintHeader is an option to ReadSoundFile. If \
set to True, the header information in the sound file (sampling rate, \
sample width, etc.) will be displayed."];

If[Not@ValueQ[PlaySoundFile::usage],PlaySoundFile::usage =
"PlaySoundFile[\"filename\"] reads the specified sound file and plays it."];

Begin["`Private`"];

{sr, sd} = 
    Switch[ $System,
        "NeXT", {44100, 16},
        "SPARC", {8000, 8},
        "Macintosh", {22254.5454, 8},
        "386", {11025, 8},
        "486", {11025, 8},
        _, {8192, 8}
    ];

Options[Waveform] = {DisplayFunction -> Identity, Overtones -> Automatic,
	PlayRange -> All, SampleDepth -> sd, SampleRate -> sr};

Options[ListWaveform] = {DisplayFunction -> Identity, PlayRange -> All,
	SampleDepth -> sd, SampleRate -> sr};

Options[AmplitudeModulation] = {DisplayFunction -> Identity,
	 PlayRange -> {-1, 1}, RingModulation -> False,
	SampleDepth -> sd, SampleRate -> sr};  

Options[FrequencyModulation] = {DisplayFunction -> Identity,
	ModulationType -> Standard, PlayRange -> All,
	SampleDepth -> sd, SampleRate -> sr};

Options[ReadSoundFile] = { PrintHeader->False } ;


(*

	Waveform

*)

Waveform::badtype = "`1` is not a valid type.";

Waveform[t_, f_?(NumberQ[N[#]]&), d_?(NumberQ[N[#]]&), opts___] :=
		With[{ out = iwf[t, f, d, opts] }, out /; out =!= $Failed ]

iwf[t_, f_, d_, opts___ ] := Module[{sr,sd,pr,id,ot},
	{sr, sd, pr, id, ot} =
        {SampleRate, SampleDepth, PlayRange, DisplayFunction, Overtones}
            /. {opts} /. Options[Waveform]; 
    wf[t, f, d, sr, sd, pr, id, ot]
]

fract[x_ ] := x - Floor[x] 

wf[type_Symbol, f_, d_,  sr_, sd_, pr_, id_, Automatic] := Module[{g,t},

    Switch[type,
        Sinusoid, g := Sin[2 Pi f t],
        Sawtooth, g := fract[-t f],
        Triangle, g := 2 Abs[fract[t f] - 1/2],
        Square, g := Sign[fract[t f] - 1/2],
        _, (Message[Waveform::badtype, type];Return[$Failed])
    ] ;

    Play[Evaluate[g], {t,0,d},
        SampleRate->sr, SampleDepth->sd, PlayRange->pr, DisplayFunction->id]
]


(*

	Fourier Summation

*)

Waveform::funder = "`1` is an insufficient number of overtones.";
Waveform::fover =
"Warning: frequency `1` Hz of highest overtone exceeds the Nyquist freqency \
`2` Hz for this sampling rate. Number of overtones reset to `3`.";
Waveform::fsine = "Warning: a sinusoid waveform has no partials.";

wf[type_Symbol, f_, d_, sr_, sd_, pr_, id_, ot_] :=
    Module[{omega, maxovertones, overtones, g, i, t},
   
    omega = N[2 Pi f] ;
    maxovertones = Floor[N[sr/2/f]];
    overtones = N[ot] ;

    If[ ( type =!= Sinusoid && overtones <= 1) || overtones == 0 ,
        (Message[Waveform::funder, overtones] ; Return[$Failed])];

    Switch[type,
        Sawtooth, If[ overtones > maxovertones, (overtones = maxovertones;
            Message[Waveform::fover, N[ot * f], sr/2.0, overtones])],
        Triangle, If[(2 overtones + 1) > maxovertones,
            (overtones = Floor[maxovertones/2];
            Message[Waveform::fover, N[(2 * ot + 1)*f], sr/2.0, overtones])],
        Square, If[(2 overtones + 1) > maxovertones,
            (overtones = Floor[maxovertones/2];
            Message[Waveform::fover, N[(2 * ot + 1)*f], sr/2.0,  overtones])],
        Sinusoid, If[ overtones > 1 , Message[Waveform::fsine]],
        _, (Message[Waveform::badtype, type];Return[$Failed])
    ];

    Switch[type,
        Sawtooth, g := Sum[1.0/i * Sin[i * omega t ], {i,1,overtones}],
        Triangle, g := Sum[1.0 / (2 i + 1)^2 * Cos[(2 i + 1) * omega t],
                        {i, 0, overtones}],
        Square, g := Sum[(1.0/(2 i + 1)) * Sin[(2 i + 1) * omega t],
                        {i,0,overtones}],
        Sinusoid, g := Sin[ omega t ]
    ];

    Play[Evaluate[g],{t,0,d},
        SampleRate->sr, SampleDepth->sd, PlayRange->pr, DisplayFunction->id]
]


wf[___] := $Failed ;




(*

	ListWaveform: Partial synthesis

*)

ListWaveform::alias = "To avoid aliasing, partial `1` will not be synthesized.";
ListWaveform::badlist = "`1` is not a list of pairs whose members are numbers.";

ListWaveform[
	p_List, f_?(NumberQ[N[#]]&), 
	d_?(NumberQ[N[#]]&), opts___] :=
		With[{ out = ilwf[p, f, d, opts] }, out /; out =!= $Failed ]

ilwf[ p_, f_, d_, opts___ ] := Module[ {sr,sd,pr,id},

	{sr, sd, pr, id} =
        { SampleRate, SampleDepth, PlayRange, 
			DisplayFunction } /. {opts} /. Options[Waveform]; 

	If[ ! And[MatrixQ[p, NumberQ[N[#]]&], Length[First[p]] == 2],
		Message[ListWaveform::badlist, p]; Return[$Failed] ];

    lwf[p, f, d, sr, sd, pr, id]
]


lwf[ p_, f_, d_, sr_, sd_, pr_, id_ ] :=
    Module[{maxpars, parlist, badpars, t},

    maxpars = Floor[N[sr/2/f]] ;
    parlist = p ;

    badpars = Select[parlist, (First[#] > maxpars)&] ;

    If[ badpars != {},
        Message[ListWaveform::alias, badpars];
        parlist = Select[parlist, (First[#] <= maxpars)&]
    ] ;

    Play[Evaluate[Plus @@ Map[(#[[2]] Sin[2 Pi #[[1]] f t])&, parlist]],
            {t,0,d}, SampleRate->sr, SampleDepth->sd,
            PlayRange->pr, DisplayFunction->id]
]

lwf[___] := $Failed ;



(* 

    Amplitude modulation
    Ring modulation

*)

AmplitudeModulation::obs =
"Warning: option Ring is obsolete, using RingModulation -> ``."

AmplitudeModulation[
	c_?((NumberQ[N[#]])&),
	m_?((NumberQ[N[#]])&),
	mi_?((NumberQ[N[#]])&),
    d_?((NumberQ[N[#]])&), 
	opts___] := With[ { out = am[c,m,mi,d,opts] }, out /; out =!= $Failed ]

am[ c_, m_, mi_, d_, opts___ ] := 
	Module[ {fc = N[2 Pi c], fm = N[2 Pi m], sr, sd, pr, id, ring, iring},
          {sr, sd, pr, id, ring} = 
            {SampleRate, SampleDepth, PlayRange, DisplayFunction,
		 RingModulation} 
                /. {opts} /. Options[AmplitudeModulation];
	  If[FreeQ[{opts}, RingModulation] && !FreeQ[{opts}, Ring],
	     ring = Ring /. {opts};
	     If[!(ring === True || ring === False),
		ring = RingModulation /. Options[AmplitudeModulation]
	     ];
	     Message[AmplitudeModulation::obs, ring]
	  ]; 
	  If[ ring == True, iring = 0.0, iring = 1.0, Return[$Failed] ];
          Play[Evaluate[ 
            (iring + mi Cos[fm t]) Cos[fc t]], {t,0,d},
                SampleRate->sr, SampleDepth->sd,
                PlayRange->pr, DisplayFunction->id]
	]

am[___] := $Failed ;




(* 

    Frequency modulation

*)

FrequencyModulation::stan = "In Standard frequency modulation, \
the second argument `1` must be a list consisting of the \
the modulating frequency and the peak deviation, both values \
measured in Hertz.";

FrequencyModulation::caspar = "In Cascade and Parallel \
frequency modulation, the second argument `1` must be a list \
of pairs, where the first member of each pair is the modulating \
frequency and the second is the peak deviation, and both values \
are measured in Hertz.";

FrequencyModulation[
    c_?((NumberQ[N[#]])&), 
    m_?((ListQ[#])&),
    d_?((NumberQ[N[#]])&), 
    opts___] := With[ { out = ifm[c,m,d,opts] }, out /; out =!= $Failed ] 

ifm[c_, m_, d_, opts___] := 
	Module[{sr,sd,pr,di,type},
    		{sr, sd, pr, di, type} = 
    			{SampleRate, SampleDepth, PlayRange, DisplayFunction,
		 	ModulationType} 
    			/.  {opts} /. Options[FrequencyModulation];
		fm[type, c, m, d, sr, sd, pr, di]
	]

fm[Standard, fc_, fm_, dur_, sr_, sd_, pr_, di_] :=
	Module[ {g,t},
        	If [ ! And[VectorQ[fm, NumberQ[#]&], Length[Flatten[fm]] === 2],
            		Message[FrequencyModulation::stan, fm];
			Return[$Failed]  ];
        	g = Sin[ 2 Pi fc t + fm[[2]] / fm[[1]] * Sin[ 2 Pi fm[[1]] t]] ;
        	Play[Evaluate[g], {t,0,dur}, SampleRate->sr, 
            		SampleDepth->sd, PlayRange->pr, DisplayFunction->di]
	]
	
fm[Cascade, fc_, fm_, dur_, sr_, sd_, pr_, di_] :=
   Module[ {g,t}, 
       If[ ! And[MatrixQ[fm, NumberQ[#]&], 
          Length[First[fm]] === 2,
          Length[fm] >= 2 ],
          Message[FrequencyModulation::caspar, fm] ; Return[$Failed] 
       ];
       g = Fold[Sin[#1 + 2 Pi #2[[1]] t] * #2[[2]]/#2[[1]] &, 0, Reverse[fm]];
       g = Sin[2 Pi fc t + g] ;
       Play[Evaluate[g], {t,0,dur}, SampleRate->sr, 
            SampleDepth->sd, PlayRange->pr, DisplayFunction->di]
   ]

fm[Parallel, fc_, fm_, dur_, sr_, sd_, pr_, di_] :=
   Module[ {g,t,i}, 
       If[ ! And[  MatrixQ[fm, NumberQ[#]&], 
          Length[First[fm]] === 2,
          Length[fm] >= 2 ],
          Message[FrequencyModulation::caspar, fm] ; Return[$Failed] 
       ];
       g = Sum[fm[[i,2]] / fm[[i,1]] Sin[2 Pi fm[[i,1]] t], {i,1,Length[fm]}];
       g = Sin[2 Pi fc t + g] ;
        Play[Evaluate[g], {t,0,dur}, SampleRate->sr, 
            SampleDepth->sd, PlayRange->pr, DisplayFunction->di]
   ]

fm[___] := $Failed ;



(*

    ReadSoundFile

*)

    
ReadSoundFile::format = "The file `1` is not in a recognized sound format.";

ReadSoundFile::wave = "The Multimedia file `1` is missing the WAVE chunk.";
ReadSoundFile::fmtchunk = "The FORMAT chunk for `1` is missing.";
ReadSoundFile::samplesize = "The sample size `1` is not supported.";
ReadSoundFile::data = "The data portion of `1` is missing.";
ReadSoundFile::tooshort = "The data section of `1` is shorter than anticipated. \
The file may be corrupt. The function will return what data could be recovered.";

ReadSoundFile::aiff = "The Apple file `1` is missing the AIFF chunk.";

showHeader[ ttype_, chan_, srate_, bits_, tbytes_, nsamps_ ] := Module[{dur},
	dur = N[nsamps / srate / chan] ;
	Print["Format: ", ttype]; 
	Print["Duration: ", dur, " seconds"];
	Print["Channels: ", chan];
	Print["Sampling rate: ", srate];
	Print["Bits per sample: ", bits];
	Print["Data size: ", tbytes, " bytes"];
	Print["Number of samples: ", nsamps];
]
	

(*

    definitions for reading and converting little- and big-endian bytes.

*)

(* littleendtoint; wave-format twos-complement little-endian ints *)
littleendtoint[a_, b_] := If[b > 127, b - 256, b] * 256 + a
littleendtoint[a_, b_, c_] := 
        (If[c > 127, c - 256, c] * 256 + b) * 256 + a
littleendtoint[a_, b_, c_, d_] := 
     ((If[d > 127, d - 256, d] * 256 + c) * 256 + b) * 256 + a

(* bigendtoint; au-format twos-complement big-endian ints *)
bigendtoint[a_, b_] := If[a > 127, a - 256, a] * 256 + b
bigendtoint[a_, b_, c_] := 
        (If[a > 127, a - 256, a] * 256 + b) * 256 + c
bigendtoint[a_, b_, c_, d_] := 
     ((If[a > 127, a - 256, a] * 256 + b) * 256 + c) * 256 + d

(* big-endian unsigned ints *)
blong[a_,b_,c_,d_] := ((((a * 256) + b) * 256) + c) * 256 + d

(* read and convert functions *)
rblong[sf_] := Module[ {b}, 
        b = ReadList[sf, Byte, 4];
		Fold[((#1 * 256) + #2)&, 0, b]
]

rbshort[sf_] := Module[ {b}, 
        b = ReadList[sf, Byte, 2];
        b[[1]] * 256 + b[[2]] 
]

rllong[sf_] := Module[ {b}, 
        b = ReadList[sf, Byte, 4];
		Fold[((#1 * 256) + #2)&, 0, Reverse[b]]
]

rlshort[sf_] := Module[ {b}, 
        b = ReadList[sf, Byte, 2];
        b[[2]] * 256 + b[[1]] 
]

(*

	readIEEE is an internal routine to convert numbers in the IEEE
	extended format.  These are 10-byte numbers, where the first two
	bytes are the exponent (excluding the top bit), and the last eight
	bytes are the mantissa (again, excluding the top bit).  The
	conversion formula comes from IEEE Standard 754 for binary
	floating-point arithmetic.

*)

readIEEE[sf_] := Module[{ieee, s, e, i, f, v},
	ieee = ReadList[sf, Byte, 10];

	(* extract top bit of exponent & exponent, top bit of mantissa & mantissa *)
	s = Quotient[ieee[[1]], 128];		
	e = (Mod[ieee[[1]], 128] * 256) + ieee[[2]] ;	
	i = Quotient[ieee[[3]], 128];		
	f = Fold[((#1 * 256) + #2)&, Mod[ieee[[3]],128],Part[ieee,Range[4,10]]]; 

	(* make the conversion based on the values extracted *)
	If[ e >= 0 && e <= 32766,
		If[ i == 1,
			v = Power[-1,s] * 2^(e-16383) * (1.0 + (f/9223372036854775808.0)),
			If[ f != 0,
				v = Power[-1,s] * 2^(e-16383) * (f/9223372036854775808.0),
				v = 0
			];
		],
		If[ e == 32767, If[ f == 0, v = Infinity, v = 0] ]
	];
	v
]

(*

    AIFF soundfile format

    One peculiarity of this format is that if a chunk has a chunksize
    with an odd number of bytes, a "pad" byte is added on to it.

*)

readAIFFsound[soundfile_, ph_] := Module[

    {sf, AIFFid, COMMid, SSNDid, chunkSize, chunkName, channels, 
    srate, samples, blockSize, bits, offset, dataSize, data, type},

    sf = soundfile ; 

	AIFFid = 1095321158 ;		(* magic numbers for chunk id's *)
	COMMid = 1129270605 ;
	SSNDid = 1397968452 ;
	type = "Apple AIFF" ;

	chunkSize = rblong[sf] ;
	chunkName = rblong[sf] ;

	If[ chunkName != AIFFid,
		Message[ReadSoundFile::aiff, sf]; Return[$Failed]];

	(*

		find the COMMON chunk that has the necessary information

	*)

	While[ chunkName != COMMid,
		chunkName = rblong[sf] ;
		chunkSize = rblong[sf] ;
		If[ chunkName != COMMid,
			If [ OddQ[chunkSize], chunkSize += 1];
			Skip[sf, Byte, chunkSize]
		]
	];

	channels = rbshort[sf];
	samples = rblong[sf];
	bits = rbshort[sf];
	srate = readIEEE[sf];

	(*

		find the SSND chunk that has the data

	*)

	While [ chunkName != SSNDid,
		chunkName = rblong[sf];
		chunkSize = rblong[sf];
		If[ chunkName != SSNDid,
			If[ OddQ[chunkSize], chunkSize += 1];
			Skip[sf, Byte, chunkSize]
		]
	];

	offset = rblong[sf];
	blockSize = rblong[sf];

	If[ offset > 0, Skip[sf, Byte, offset]];

	dataSize = chunkSize - offset - 8;

   (* kludge to hook up PlaySoundFile *)
    $InternalSampleRate = srate;
   (* display information *)
	If[ph, showHeader[type, channels, srate, bits, dataSize, samples]];

	(*

		convert the data into 16-bit signed numbers before shipping it
		back

	*)

    data = ReadList[sf, Byte, dataSize];

	Switch [ bits,
    	8, data = Map[ (If[# > 127, # - 256, #] * 256)&, data ], 
 		16, data = Apply[ If[ #1 > 127, ( #1 - 256 ) * 256 + #2,
			( #1 * 256 ) + #2]&, Partition[data,2],{1}],
		_, (Message[ReadSoundFile::samplesize, bits] ; Return[$Failed])
	];

    If[ channels == 2, 
            Transpose[Partition[data,2]],
            data
    ]
]

(*

    Windows Multimedia WAVE files

*)

readWaveSound[soundfile_, ph_] := Module[

    {sf, WAVEid, FMTid, DATAid, fmtlength, format, channels, 
    samplingRate, avgBytesPerSec, bytesPerSample, bitsPerSample, dataSize,
    data, chunklength, tmpid, scalefactor, bytesPerChannel},

    sf = soundfile ; 

    WAVEid = 1163280727 ;     (* magic numbers for WAVE files *)
    FMTid = 544501094 ;
    DATAid = 1635017060 ;
	
    Skip[sf,Byte,4];	(* skip sizeof file *)

    If [ WAVEid != rllong[sf],
        (Message[ReadSoundFile::wave, sf];Return[$Failed];)];
   (* try to find the format chunk; exit if it can't be found *)
    While[ FMTid =!= (tmpid = rllong[sf]),
        If[!FreeQ[tmpid, EndOfFile],
            Message[ReadSoundFile::fmtchunk, sf];Return[$Failed]
        ];
        chunklength = rllong[sf];
        If[OddQ[chunklength], chunklength++];
        Skip[sf, Byte, chunklength]
    ];
 (*
    If [ FMTid != rllong[sf],
        (Message[ReadSoundFile::fmtchunk, sf];Return[$Failed];)];
*)
    fmtlength = rllong[sf];
    If[OddQ[fmtlength], fmtlength++];
    
    Switch[ rlshort[sf],  (* this is written to enable other format types eventually *)
        1, format = "Microsoft PCM WAVE RIFF",
        _, Message[ReadSoundFile::format, sf]; Return[$Failed]
    ] ;

    channels = rlshort[sf];
    samplingRate = rllong[sf];
    avgBytesPerSec = rllong[sf];
    bytesPerSample = rlshort[sf];
	bitsPerSample = rlshort[sf];
	
	bytesPerChannel = bytesPerSample/channels;
	
   Skip[sf, Byte, (fmtlength - 16)];

   (* try to find the data chunk; exit if it can't be found *)
    While[ DATAid =!= (tmpid = rllong[sf]),
        If[!FreeQ[tmpid, EndOfFile],
            Message[ReadSoundFile::data, sf];Return[$Failed]
        ];
        chunklength = rllong[sf];
        If[OddQ[chunklength], chunklength++];
        Skip[sf, Byte, chunklength]
    ];
    
    (* now in data chunk; find size of data *)
    dataSize = rllong[sf];
    
    (* calculate number of samples *)
    samples = dataSize/bytesPerSample;
    
    (* kludge to hook up PlaySoundFile *)
    $InternalSampleRate = samplingRate;
   (* display information *)
	If[ph, 
		showHeader[format,channels,samplingRate,bitsPerSample,dataSize,samples]];

   (* read data *)
    data = ReadList[sf, Byte, dataSize];
   (* correct if not enough data in file *)
    If[Last[data] === EndOfFile,
       Message[ReadSoundFile::tooshort, First[sf]];
       data = Take[data, First[First[Postion[data, EndOfFile]]] - 1 ]
    ];
   (* convert wave data (little endian bytes, twos-comp in the case 9 or more
      bits per sample, unsigned at 8 bits or less) to ints *)
    If[bytesPerChannel =!= 1,
       data = Apply[ littleendtoint,
             Partition[data, bytesPerChannel],
           {1}]
     ];
    
    (* calculate factor to scale data to 2^16; if not integer, we don't want
       fractional values returned, but instead floats, I think; I am
       anticipating 3 or 4 bytes per channel, but I think the spec only
       covers 1 or 2. Note that WAVE format scales unused bits to the
       least-significant bits, so account for bytesPerChannel, not
       bitsPerChannel *)
    scalefactor = 2^16/(2^(8 * bytesPerChannel));
    If[Not[IntegerQ[scalefactor]], scalefactor = N[scalefactor]];
    
   (* rescale *)
    If[bytesPerChannel === 1,
        data = (data - 128) * scalefactor, (* rem: 8- bits data unsigned... *)
        data = data * scalefactor          (* ... while 9+ bits signed *)
    ];
    (* split if multi-channel *)
    If[channels === 1,
        data,
        Transpose[Partition[data, channels]]
    ]
]


(*

    NeXT/SUN soundfiles

*)

mulaw2linear = {-32124, -31100, -30076, -29052, -28028, -27004, -25980,
    -24956, -23932, -22908, -21884, -20860, -19836, -18812, -17788,
    -16764, -15996, -15484, -14972, -14460, -13948, -13436, -12924,
    -12412, -11900, -11388, -10876, -10364, -9852, -9340, -8828, -8316,
    -7932, -7676, -7420, -7164, -6908, -6652, -6396, -6140, -5884,
    -5628, -5372, -5116, -4860, -4604, -4348, -4092, -3900, -3772,
    -3644, -3516, -3388, -3260, -3132, -3004, -2876, -2748, -2620,
    -2492, -2364, -2236, -2108, -1980, -1884, -1820, -1756, -1692,
    -1628, -1564, -1500, -1436, -1372, -1308, -1244, -1180, -1116,
    -1052, -988, -924, -876, -844, -812, -780, -748, -716, -684, -652,
    -620, -588, -556, -524, -492, -460, -428, -396, -372, -356, -340,
    -324, -308, -292, -276, -260, -244, -228, -212, -196, -180, -164,
    -148, -132, -120, -112, -104, -96, -88, -80, -72, -64, -56, -48,
    -40, -32, -24, -16, -8, 0, 32124, 31100, 30076, 29052, 28028, 27004,
    25980, 24956, 23932, 22908, 21884, 20860, 19836, 18812, 17788,
    16764, 15996, 15484, 14972, 14460, 13948, 13436, 12924, 12412,
    11900, 11388, 10876, 10364, 9852, 9340, 8828, 8316, 7932, 7676,
    7420, 7164, 6908, 6652, 6396, 6140, 5884, 5628, 5372, 5116, 4860,
    4604, 4348, 4092, 3900, 3772, 3644, 3516, 3388, 3260, 3132, 3004,
    2876, 2748, 2620, 2492, 2364, 2236, 2108, 1980, 1884, 1820, 1756,
    1692, 1628, 1564, 1500, 1436, 1372, 1308, 1244, 1180, 1116, 1052,
    988, 924, 876, 844, 812, 780, 748, 716, 684, 652, 620, 588, 556,
    524, 492, 460, 428, 396, 372, 356, 340, 324, 308, 292, 276, 260,
    244, 228, 212, 196, 180, 164, 148, 132, 120, 112, 104, 96, 88, 80,
    72, 64, 56, 48, 40, 32, 24, 16, 8, 0};

readNeXTSunSound[soundfile_, ph_] := Module[

    {dataLocation, dataSize, dataFormat, samplingRate, dataType,
    channelCount, data, sf, header, samples, bytes, ttype,
	sizeofheader},

    sf = soundfile ;
	ttype = "NeXT/Sun" ;
	sizeofheader = 24 ; (* minimum size of NeXT/SUN soundfile header *)

    header = ReadList[sf,Byte,20]; 

    (*
	{dataLocation, dataSize, dataFormat, samplingRate, channelCount} =
        Apply[blong, Partition[header,4],{1}];
	*)

    {dataLocation, dataSize, dataFormat, samplingRate, channelCount} =
        Apply[(((((#1 * 256) + #2) * 256) + #3) * 256 + #4)&,
			Partition[header,4],{1}];

	infoLength = dataLocation - 24 ;

    {dataType, bytes} =
        Switch[ dataFormat,
            1, {"8-bit mulaw", 1},
            2, {"8-bit linear", 1},
            3, {"16-bit linear", 2},
            4, {"24-bit linear", 3},
            5, {"32-bit linear", 4},
            6, {"float", 4},
            7, {"double", 8},
            8, {"indirect", 1},
            9, {"nested", 1},
            10, {"DSP core ", 1},
            11, {"DSP data 8", 1},
            12, {"DSP data 16", 1},
            13, {"DSP data 24", 1},
            14, {"DSP data 32", 1},
            16, {"display", 1},
            17, {"mulaw squelch", 1},
            18, {"emphasized", 1},
            19, {"compressed", 1},
            20, {"compressed-emphasized", 1},
            21, {"DSP commands", 1},
            22, {"DSP commands samples", 1}
        ];
            
    (* number of samples *)
    samples = Floor[N[dataSize / bytes / channelCount]] ;
    
   (* kludge to hook up PlaySoundFile *)
    $InternalSampleRate = samplingRate;
   (* display information, otherwise skip over info field *)

	If[ph, 
		showHeader[ttype,channelCount,samplingRate,(bytes*8),dataSize,samples];
		If[ MemberQ[{1,2}, dataFormat], Print["Encoding: ", dataType]];
		Print["Text: ", If[infoLength === 0, "",
		       FromCharacterCode[DeleteCases[
                           ReadList[sf,Byte,infoLength], 0]]]],
		Skip[sf, Byte, infoLength]
	];

    (* Return if format is not mu-law or 16-bit linear *)

	If[ ! MemberQ[{1,2,3,4,5}, dataFormat], 
        Message[ReadSoundFile::format, sf] ; Return[$Failed]
	];

    data = ReadList[sf,Byte];

    (* Convert bigendian data to signed ints, rescaling if necessary
       to the expected +/- 2^15 range; output floats for 24 or 32 bit *)

    Switch[ dataFormat,
        1, data = Table[mulaw2linear[[(data[[x]]+1)]],{x,1,samples}],
        2, data = Map[If[# > 127, # - 256, #]&, data] * 256,
        3 | 4 | 5, data = Apply[ bigendtoint,
                    Partition[data,bytes],{1}];
                   If[bytes > 2, data = data * N[2^16/(2^(8 * bytes))]]
    ] ;
    
    (* if the soundfile is stereo, return two lists *)
    
    If[ channelCount == 2, 
            Transpose[Partition[data,2]],
            data
    ]
]


(*

	main routine for ReadSoundFile

*)

ReadSoundFile[name_String, opts___] := With[{ out = rsf[name, opts] }, out ]

rsf[name_, opts___] := Module[

    {sf, header, NeXTSunID, RIFFid, little, big, out, printheader},

    NeXTSunID = 779316836 ;     (* magic number for NeXT/Sun soundfiles *)
    RIFFid = 1179011410 ;       (* magic number for Windows RIFF files *)
	AppleID = 1179603533 ;		(* magic number for Apple AIFF files *)

    {printheader} = {PrintHeader} /. {opts} /. Options[ReadSoundFile];

    sf = OpenRead[name, BinaryFormat -> True] ;    (* quit if file not found *)
    If[SameQ[sf, $Failed], Return[$Failed]];
    
    header = ReadList[sf,Byte,4];

	little = Fold[ ((#1 * 256) + #2)&, 0, Reverse[header]];
	big = Fold[ ((#1 * 256) + #2)&, 0, header];

    If[ little == RIFFid, out = readWaveSound[sf, printheader],
		Switch[ big,
			NeXTSunID, out = readNeXTSunSound[sf, printheader],
			AppleID, out = readAIFFsound[sf, printheader],
            _, Message[ReadSoundFile::format, name]
        ]
    ] ;

    Close[sf];

    out 
]

Options[PlaySoundFile] = Append[Options[ListPlay], PrintHeader -> False];

SetOptions[PlaySoundFile,
    PlayRange -> {-2^15, 2^15},
    SampleRate -> Automatic
];

(* PlaySoundFile; uses a kludge to link up the sampling rate to
   ReadSoundFile, plays a sound. Use DisplayFunction -> Identity
   to suppress sound output. *)
PlaySoundFile[filename_String, opts___?OptionQ] :=
    Module[{dat, rate},
        dat = ReadSoundFile[filename,
            FilterRules[##, Options[ReadSoundFile]]& @@
               Flatten[{opts,Options[PlaySoundFile]}] ];
        If[!ListQ[dat], Return[dat]];
        {rate} = {SampleRate}/.Flatten[{opts, Options[PlaySoundFile]}];
        If[rate === Automatic,
            rate = $InternalSampleRate
        ];
        ListPlay[dat,
                FilterRules[##, Options[ListPlay]]& @@
                   Flatten[{SampleRate -> rate, opts,
                    Options[PlaySoundFile]}]
        ]
    ]

(* Obsolete function name *)
ReadSoundfile = ReadSoundFile;

(*

        Protect user-accessible functions.

*)

Protect[ Waveform, AmplitudeModulation, FrequencyModulation, ReadSoundFile ]; 

End[];

EndPackage[];
