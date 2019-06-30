pragma solidity ^0.4.24;

import "./Chainlinked.sol";

contract MyContract is Chainlinked {

	uint256 public currentPrice = 0;
	address public owner;

	constructor() public{
		setLinkToken(0x20fE562d797A42Dcb3399062AE9546cd06f63280);
		setOracle(0x261a3F70acdC85CfC2FFc8badE43b1D42bf75D69);
		owner = msg.sender;
	}

	function requestEthereumPrice(string _currency) public returns (bytes32 requestId) {
		ChainlinkLib.Run memory run = newRun(bytes32("2216dd2bf5464687a05ded0b844e200c"), this, "fulfillEthereumPrice(bytes32,uint256)");
		run.add("url", "https://min-api.cryptocompare.com/data/price?fsym=ETH&tsyms=USD,EUR,JPY");
		string[] memory path = new string[](1);
		path[0] = _currency;
		run.addStringArray("path", path);
		run.addInt("times", 100);
		requestId = chainlinkRequest(run, LINK(1));
	}

	function fulfillEthereumPrice(bytes32 _requestId, uint256 _price) public checkChainlinkFulfillment(_requestId) {
		currentPrice = _price;
	}

}




