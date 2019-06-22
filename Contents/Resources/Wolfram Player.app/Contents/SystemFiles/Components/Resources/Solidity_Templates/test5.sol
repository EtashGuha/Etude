pragma solidity ^0.4.23;

contract test1{
	
	string public name;
	string public version;
	uint256 public supply;	

	constructor(string _name, string _version, uint256 _supply) public {
		name = _name;
		version = _version;
		supply = _supply;
	}

}

contract test2{
	
	string public name2;
	string public version2;
	uint256 public supply2;	

	constructor(string _name, string _version, uint256 _supply) public {
		name2 = _name;
		version2 = _version;
		supply2 = _supply;
	}

}