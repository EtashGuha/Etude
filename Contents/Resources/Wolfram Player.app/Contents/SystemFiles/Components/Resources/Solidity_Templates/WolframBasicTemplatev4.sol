pragma solidity ^0.4.23;

contract WolframBasicTemplate {
	string Storage;
	string StorageAddress;
	string FileHash;
	string ResultAddress;
	bool Status;
	bool PaymentStatus;
	address Owner; /// User Account
	address Receiver;
	string WServerToken; /// Encoded Token with Private Key
	event logWolframSM(string indexed token);
	
	constructor(string _Storage, string _StorageAddress, string _FileHash, string _WServerToken, address _Receiver) public payable {
	    Storage = _Storage;
	    StorageAddress = _StorageAddress;
	    FileHash = _FileHash;
	    ResultAddress = "";
	    Status = false;
	    PaymentStatus = false;
	    Owner = msg.sender;
	    Receiver = _Receiver;
	    WServerToken = _WServerToken;
	    emit logWolframSM(_WServerToken);
	}

	function getStorage() public constant returns (string){
	    return Storage;
	}	

	function getStorageAddress() public constant returns (string){
	    return StorageAddress;
	}

	function getFileHash() public constant returns (string){
		return FileHash;
	}
    
    function getResultAddress() public constant returns (string){
	    return ResultAddress;
	}
	
	function getStatus() public constant returns (bool){
	    return Status;
	}

	function getOwner() public constant returns (address){
		return Owner;
	}

	function getReceiver() public constant returns (address){
		return Receiver;
	}

	function getBalance() constant public returns(uint) {
        return address(this).balance;
    }

    function getWToken() constant public returns(string){
    	return WServerToken;
    }

	function getPaymentStatus() constant public returns(bool){
    	return PaymentStatus;
    }

    function depositFunds() public payable {   
    }

	function setResultAddress(string _ResultAddress) public {
        ResultAddress = _ResultAddress;
        Status = true;
    }

    /*function setPaymentResult(string _ResultAddress, bool _Pay) public {
    	ResultAddress = _ResultAddress;
    	Status = true; //This is ExecuteStatus
    	uint amount = address(this).balance;
    	if(_Pay){
    		Receiver.transfer(amount);
    		PaymentStatus = true;
    	} else {
    		PaymentStatus = false;
    	}
    }*/

    function setPaymentResult(string _ResultAddress) public {
    	ResultAddress = _ResultAddress;
    	Status = true; //This is ExecuteStatus
    	uint amount = address(this).balance;
    	Receiver.transfer(amount);
    	PaymentStatus = true;
    	
    }
	

}