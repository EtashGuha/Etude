pragma solidity ^0.4.23;

contract WolframBasicTemplate {
	string public Content;
	string public TemplateVersion = "v2";
	string public WServerToken; // Encoded Token to identify wolfram contracts
	string public Result;
	bool public Evaluated;
	
	event logWolframSM(string indexed token);
	
	constructor(string _WServerToken, string _Content) public {
	    WServerToken = _WServerToken;
	    Content = _Content;
	    Result = "";
	    Evaluated = false;
	    emit logWolframSM(_WServerToken);
	}

	function setResult(string _Result) public {
        Result = _Result;
        Evaluated = true;
    }

}