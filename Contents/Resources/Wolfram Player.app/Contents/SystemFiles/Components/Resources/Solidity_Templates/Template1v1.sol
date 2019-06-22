pragma solidity ^0.4.23;

contract WolframBasicTemplate {
	string public ExternalStorage;
	string public ExternalStorageAddress;
	string public FileHash;
	string public FileName;
	string public TemplateVersion = "v1";
	string public WServerToken; /// Encoded Token to identify Wolfram contracts
	string public ResultAddress;
	bool public Evaluated;
	
	event logWolframSM(string indexed token);
	
	constructor(string _WServerToken, string _ExternalStorage, string _ExternalStorageAddress, string _FileName,string _FileHash) public {
	    WServerToken = _WServerToken;
	    ExternalStorage = _ExternalStorage;
	    ExternalStorageAddress = _ExternalStorageAddress;
	    FileHash = _FileHash;
	    FileName = _FileName;
	    ResultAddress = "";
	    Evaluated = false;
	    emit logWolframSM(_WServerToken);
	}

	function setResultAddress(string _ResultAddress) public {
        ResultAddress = _ResultAddress;
        Evaluated = true;
    }

}