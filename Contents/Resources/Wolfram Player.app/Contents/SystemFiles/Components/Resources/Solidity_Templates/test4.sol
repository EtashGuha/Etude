pragma solidity ^0.4.23;

contract inputs{
	
	string public name;
	string public version;
	uint256 public supply;	

	constructor(string _name, string _version, uint256 _supply) public {
		name = _name;
		version = _version;
		supply = _supply;
	}

	function sum(uint256 a, uint256 b) public pure returns (uint256){
		return a + b;
	}

}