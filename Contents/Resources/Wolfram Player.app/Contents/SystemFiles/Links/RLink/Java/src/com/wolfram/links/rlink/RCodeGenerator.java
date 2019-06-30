package com.wolfram.links.rlink;

/**
 * 
 * RLinkJ source code (c) 2011-2012, Wolfram Research, Inc.
 * 
 * 
 * 
 * This file is part of RLinkJ interface to JRI Java library.
 * 
 * RLinkJ is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 2 of the License, or (at your option) any
 * later version.
 * 
 * RLinkJ is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with RLinkJ. If not, see <http://www.gnu.org/licenses/>.
 * 
 * 
 * 
 * 
 * @author Leonid Shifrin
 * 
 */

public class RCodeGenerator {

	public static String rGetAttributeCode(String rVarName, String attName) {
		return "attr(" + rVarName + ",\"" + attName + "\")";
	}

	public static String rGetAllAttributesCode(String rVarName) {
		return "attributes(" + rVarName + ")";
	}

	public static String rGetType(String objName) {
		return "typeof(" + objName + ")";
	}

	public static String rGetClass(String objName) {
		return "class(" + objName + ")";
	}

	public static String rGetAttributeType(String rVarName, String attName) {
		return rGetType(rGetAttributeCode(rVarName, attName));
	}

	public static String listIndex(String var, int ind) {
		return var + "[[" + new Integer(ind).toString() + "]]";
	}

	public static String getTemporaryVariable() {
		return "myRandomVar1234567"
				+ new Integer((int) (Math.random() * 100000));
	}

}
