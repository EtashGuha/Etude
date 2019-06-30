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

import java.util.Arrays;

import com.wolfram.links.rlink.RExecutor;

public class RLogicalVectorInType extends RVectorInType {

	private final boolean[] elements;

	public RLogicalVectorInType(boolean[] elements,
			int[] missingElementPositions, IRInAttributes attributes) {
		super(missingElementPositions, attributes);
		this.elements = elements;
	}

	public boolean[] getElements() {
		return elements;
	}

	@Override
	public String toString() {
		return "RLogicalVectorInType [elements=" + Arrays.toString(elements)
				+ ", attributes=" + attributes + "]";
	}

	@Override
	public boolean rPut(String rVar, RExecutor exec) {
		return exec.assignBooleanArray(rVar, this.elements)
				&& super.rPut(rVar, exec);
	}

}
