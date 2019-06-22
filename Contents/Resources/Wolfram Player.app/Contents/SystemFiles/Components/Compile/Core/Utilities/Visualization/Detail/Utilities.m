
fastRasterize[expr_Cell, format_String, alphaChannel:(True|False):False] := Module[{boundingBox, baseline, imgWidth, imgHeight, packetResult},
    packetResult=UsingFrontEnd[MathLink`CallFrontEnd[ExportPacket[expr,format, Verbose->True, AlphaChannel->alphaChannel]]];
    boundingBox="OutputBoundingBox"/.Last[packetResult];
    baseline=Round[Baseline /. Last[packetResult]];
    {imgWidth,imgHeight}=Round[Subtract@@@(Reverse/@boundingBox)];
    {{imgWidth, imgHeight, baseline}, First[packetResult]}
]

