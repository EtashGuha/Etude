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

public abstract class RNumericVectorOutType extends RVectorOutType implements
		IROutNumericVector {

	private int[] NaNElementPositions = new int[0];
	private int[] positiveInfinityPositions = new int[0];
	private int[] negativeInfinityPositions = new int[0];
	private int[] complexInfinityPositions = new int[0];

	public RNumericVectorOutType(String expr) {
		super(expr);
	}

	@Override
	public int[] getNaNElementPositions() {
		return this.NaNElementPositions;
	}

	@Override
	public boolean rGet(RExecutor exec) {
		final String code = this.getVariableNameOrCodeString();
		this.NaNElementPositions = exec.getNaNElementPositions(code);
		this.positiveInfinityPositions = exec
				.getPositiveInfinityPositions(code);
		this.negativeInfinityPositions = exec
				.getNegativeInfinityPositions(code);
		this.complexInfinityPositions = exec.getComplexInfinityPositions(code);
		return super.rGet(exec);
	}

	@Override
	public int[] getPositiveInfinityPositions() {
		return this.positiveInfinityPositions;
	}

	@Override
	public int[] getNegativeInfinityPositions() {
		return this.negativeInfinityPositions;
	}

	@Override
	public int[] getComplexInfinityPositions() {
		return this.complexInfinityPositions;
	}

}
