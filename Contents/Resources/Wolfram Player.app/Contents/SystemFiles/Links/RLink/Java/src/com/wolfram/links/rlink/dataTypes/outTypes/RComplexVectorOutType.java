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

import java.util.Arrays;

import com.wolfram.links.rlink.RExecutor;
import com.wolfram.links.rlink.dataTypes.auxiliary.Complex;

public class RComplexVectorOutType extends RNumericVectorOutType {

	private Complex[] elements = new Complex[0];

	public RComplexVectorOutType(String expr) {
		super(expr);
	}

	@Override
	public boolean rGet(RExecutor exec) {
		double[] re = exec.evalGetDoubleArray("Re("
				+ this.getVariableNameOrCodeString() + ")");
		double[] im = exec.evalGetDoubleArray("Im("
				+ this.getVariableNameOrCodeString() + ")");
		if (re == null || im == null) {
			return false;
		}
		Complex[] result = new Complex[re.length];
		for (int i = 0; i < result.length; i++) {
			result[i] = new Complex(re[i], im[i]);
		}
		this.elements = result;
		return super.rGet(exec);
	}

	@Override
	public ROutTypes getType() {
		return ROutTypes.COMPLEX_VECTOR;
	}

	@Override
	public Complex[] getElements() {
		return this.elements;
	}

	@Override
	public String toString() {
		return "RComplexVectorOutType [elements=" + Arrays.toString(elements)
				+ ", attributes=" + attributes + ", getNaNElementPositions()="
				+ Arrays.toString(getNaNElementPositions())
				+ ", getMissingElementPositions()="
				+ Arrays.toString(getMissingElementPositions()) + "]";
	}

}
