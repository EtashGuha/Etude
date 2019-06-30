package com.wolfram.links.rlink.dataTypes.outTypes;

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
import com.wolfram.links.rlink.RLinkInit;

public abstract class RFunctionOutType extends ROutTypeImpl {

	protected static int maxFuncRefNumber = 0;
	private int refNumber = 0;
	private RDeparsedCodeOutType deparsedFunctionSource;

	public RFunctionOutType(String expr) {
		super(expr);
		this.deparsedFunctionSource = new RDeparsedCodeOutType(expr);
	}

	public static String getFunctionHashElement(int ref) {
		return RLinkInit.FUNCTION_HASH_VAR_NAME + "[[" + ref + "]]";
	}

	public int getRefNumber() {
		return refNumber;
	}

	public RDeparsedCodeOutType getDeparsedFunctionSource() {
		return deparsedFunctionSource;
	}

	public static boolean isValidFunctionReference(int ref, RExecutor exec) {
		if (!(ref > 0 && ref <= maxFuncRefNumber)) {
			return false;
		}
		String type = exec.getRObjectType(getFunctionHashElement(ref));
		return ROutTypes.CLOSURE.getStringType().equals(type)
				|| ROutTypes.BUILTIN.getStringType().equals(type);
	}

	@Override
	public boolean rGet(RExecutor exec) {
		boolean result = exec.eval(getFunctionHashElement(++maxFuncRefNumber)
				+ " <- " + this.getVariableNameOrCodeString());
		if (!result) {
			maxFuncRefNumber--;
			return false;
		}
		deparsedFunctionSource.rGet(exec);
		refNumber = maxFuncRefNumber;
		return super.rGet(exec);
	}
}
