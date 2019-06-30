pragma solidity ^0.4.23;

contract Test2{

	function sum(uint256 a, uint256 b) public pure returns (uint256){
		return a + b;
	}

}

contract Test is Test2{

    function main(uint256 a, uint256 b) public pure returns (uint256){
        return sum(a,b);
    }

}