pragma solidity ^0.4.23;

import "./test2.sol";

contract Test is Test2{

    function main(uint256 a, uint256 b) public pure returns (uint256){
        return sum(a,b);
    }

}