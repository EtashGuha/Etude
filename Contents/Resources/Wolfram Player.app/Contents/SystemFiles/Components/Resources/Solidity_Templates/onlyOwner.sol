pragma solidity ^0.4.23;

contract Test {

	string public version;
	address public owner;

	constructor() public {
		owner = msg.sender;
	}

	modifier onlyOwner {
        require(
            msg.sender == owner,
            "Only owner can call this function."
        );
        _;
    }

	function setVersion(string _version) public onlyOwner returns(string) {
        version = _version;
        return "Function called succesfully!.";
    }

}