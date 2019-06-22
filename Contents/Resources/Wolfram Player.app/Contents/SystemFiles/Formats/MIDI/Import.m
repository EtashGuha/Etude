(* ::Package:: *)

Begin["System`Convert`MIDIDump`"];


ImportExport`RegisterImport["MIDI",
{
  (* Sound elements *)
  {"Sound"}            :>     (ImportMIDISound[All][##]&),
  {"Sound", i_Integer} :>     (ImportMIDISound[i  ][##]&),
  {"SoundNotes"}            :> (ImportMIDIData[All  ,"SoundNotes"][##]&),
  {"SoundNotes", i_Integer} :> (ImportMIDIData[i,   "SoundNotes"][##]&),

  (* metainfo elements *)
  {"Duration"} :>   (ImportMIDIData[All,"Duration"][##]&),  
  {"Metadata"}            :> (ImportMIDIData[All,"Metadata"][##]&),
  {"Metadata", i_Integer} :> (ImportMIDIData[i  ,"Metadata"][##]&),
  "TrackCount" :> ImportMIDITrackCount,
  "Header" :> ImportMIDIHeader,

  (* raw data *)
  {"RawData"}            :> (ImportMIDIData[All  ,"RawData"][##]&),
  {"RawData", i_Integer} :> (ImportMIDIData[i,    "RawData"][##]&),

  (* default *)
  ImportMIDIHeader
},
  "AvailableElements" -> {"Duration", "Header", "Metadata", "RawData", "Sound", "SoundNotes", "TrackCount"},
  "BinaryFormat" -> True,
  "DefaultElement" -> "Sound",
  "FunctionChannels" -> {"Streams"}
]


End[];
