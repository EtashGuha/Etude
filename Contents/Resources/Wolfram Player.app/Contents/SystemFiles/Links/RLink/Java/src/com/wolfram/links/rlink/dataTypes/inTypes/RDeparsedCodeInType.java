package com.wolfram.links.rlink.dataTypes.inTypes;

/**
 * 
 *  RLinkJ source code (c) 2011-2012, Wolfram Research, Inc. 
 *  
 *  
 *  
 *   This file is part of RLinkJ interface to JRI Java library.
 *
 *   RLinkJ is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU Lesser General Public License as 
 *   published by the Free Software Foundation, either version 2 of 
 *   the License, or (at your option) any later version.
 *
 *   RLinkJ is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU Lesser General Public 
 *   License along with RLinkJ. If not, see <http://www.gnu.org/licenses/>.
 *  
 * 
 * 
 *
 * @author Leonid Shifrin
 *
 */

import com.wolfram.links.rlink.RExecutor;

public class RDeparsedCodeInType extends RInTypeImpl {

	private String[] deparsedCodeParts;
	private boolean evalCode;

	public RDeparsedCodeInType(String[] deparsedCodeParts,
			IRInAttributes attributes, Boolean evalCode) {
		super(attributes);
		this.deparsedCodeParts = deparsedCodeParts;
		this.evalCode = evalCode;
	}

	@Override
	public boolean rPut(String rVar, RExecutor exec) {
		String tempVar = exec.getRandomVariable();
		RCharacterVectorInType partArray = new RCharacterVectorInType(
				deparsedCodeParts, new int[0], new RInAttributesImpl());
		boolean result = partArray.rPut(tempVar, exec);
		if (!result)
			return false;
		String code = rVar + " <- " + (evalCode ? "eval(" : "")
				+ "parse(text = " + tempVar + ")" + (evalCode ? ")" : "");
		result = exec.eval(code);
		exec.removeRVariable(tempVar);
		return result && super.rPut(rVar, exec);
	}

}
