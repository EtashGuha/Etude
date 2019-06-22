(* ::Package:: *)

Begin["System`Convert`PDFDump`"]


ImportExport`RegisterImport[
  "PDF",
  {
    "Pages" 							:> PDFImportFile,
	"Plaintext" 						:> PDFImportFileAsText,
	"PageCount" 						:> PDFPageCount,
	"Attachments" 						:> PDFAttachments,
	"RawAttachments"					:> PDFRawAttachments,
	"Images"							:> PDFImages,
    {"Pages", n_Integer?Positive}		:> PDFImportPage[n],
	{"Plaintext", n_Integer?Positive}	:> PDFImportPageAsText[n],
	{"Pages", "Elements"}				:> PDFLength["Pages"],
	{"Plaintext", "Elements"}			:> PDFLength["Plaintext"],
	PDFInformation
 },
 {},
 "Sources" -> {"Convert`PDF`", "PDF.exe"},
 "AvailableElements" -> {"Attachments", "Author", "CreationDate", "Creator", "Images", 
			"Keywords", "PageCount", "Pages", "Plaintext", "RawAttachments", "Title"},
 "DefaultElement" -> "Pages",
 "BinaryFormat" -> True
]


End[]
