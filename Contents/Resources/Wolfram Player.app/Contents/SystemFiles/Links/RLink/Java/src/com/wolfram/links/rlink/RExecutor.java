package com.wolfram.links.rlink;

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

import java.util.HashMap;

import org.rosuda.JRI.RBool;
import org.rosuda.JRI.REXP;
import org.rosuda.JRI.Rengine;

import com.wolfram.links.rlink.exceptions.RLinkErrorException;

public class RExecutor {
	private Rengine engine = null;

	RExecutor(Rengine engine) {
		this.engine = engine;
	}

	// TODO: handle errors!
	public synchronized String[] getAttributeNames(String rVarName) {
		REXP x = engine.eval(RCodeGenerator.rGetAllAttributesCode(rVarName));
		if (x == null) {
			return null;
		}
		return x.asList().keys();
	}

	private synchronized REXP evalAndCheck(String code) {
		REXP result = engine.eval(code);
		if (result == null) {
			throw new RLinkErrorException(
					"The error was encountered while executing the following code in R:"
							+ code);
		}
		return result;
	}

	public synchronized boolean eval(String code) {
		return engine.eval(code) != null;
	}

	public int getLength(String rVar) {
		return evalAndCheck("length(" + rVar + ")").asInt();
	}

	public String evalGetString(String code) {
		return evalAndCheck(code).asString();
	}

	public String[] evalGetStringArray(String code) {
		return evalAndCheck(code).asStringArray();
	}

	public double[] evalGetDoubleArray(String code) {
		return evalAndCheck(code).asDoubleArray();
	}

	public int[] evalGetIntArray(String code) {
		return evalAndCheck(code).asIntArray();
	}

	// TODO: have to rewrite to account for NA-s once I support them
	public boolean[] evalGetBooleanArray(String code) {
		int[] v = evalGetIntArray(code);
		// This is needed because of the bug of REXP.asIntArray method for NA
		// boolean elements (should return 2 fir them, returns min negative int
		int[] missing = getMissingElementPositions(code);
		HashMap<Integer, Boolean> missingHash = new HashMap<Integer, Boolean>();
		for (int i : missing) {
			missingHash.put(i, true);
		}
		boolean[] result = new boolean[v.length];
		for (int i = 0; i < v.length; i++) {
			switch (v[i]) {
			case 0:
			case 2: // NA
				result[i] = false;
				break;
			case 1:
				result[i] = true;
				break;
			default:
				if (missingHash.containsKey(i + 1)) {
					// Work around the bug in REXP.asIntArray
					result[i] = false;
				} else {
					System.out.println("Unknown element : " + v[i]);
					throw new RLinkErrorException(
							"Could not convert a result to a boolean vector");
				}
			}
		}
		return result;
	}

	public synchronized boolean assignStringArray(String expr, String[] array) {
		String var = getRandomVariable();
		if (!engine.assign(var, array)) {
			return false;
		}
		if (!eval(expr + " <- " + var)) {
			return false;
		}
		if (!removeRVariable(var)) {
			return false;
		}
		return true;
	}

	public synchronized boolean assignDoubleArray(String expr, double[] array) {
		String var = getRandomVariable();
		if (!engine.assign(var, array)) {
			return false;
		}
		if (!eval(expr + " <- " + var)) {
			return false;
		}
		if (!removeRVariable(var)) {
			return false;
		}
		return true;
	}

	public synchronized boolean assignIntArray(String expr, int[] array) {
		String var = getRandomVariable();
		if (!engine.assign(var, array)) {
			return false;
		}
		if (!eval(expr + " <- " + var)) {
			return false;
		}
		if (!removeRVariable(var)) {
			return false;
		}
		return true;
	}

	public synchronized boolean assignBooleanArray(String expr, boolean[] array) {
		String var = getRandomVariable();
		if (!engine.assign(var, array)) {
			return false;
		}
		if (!eval(expr + " <- " + var)) {
			return false;
		}
		if (!removeRVariable(var)) {
			return false;
		}
		return true;
	}

	public boolean assignComplexArray(String expr, double[] re, double[] im) {
		String reVar = getRandomVariable();
		String imVar = getRandomVariable();
		if (!assignDoubleArray(reVar, re)) {
			return false;
		}
		if (!assignDoubleArray(imVar, im)) {
			removeRVariable(reVar);
			return false;
		}
		if (!eval(expr + " <- " + reVar + " + 1i * " + imVar)) {
			return false;
		}
		if (!removeRVariable(reVar) || !removeRVariable(imVar)) {
			return false;
		}
		return true;
	}

	public boolean isNULL(String var) {
		String type = this.evalGetString(RCodeGenerator.rGetType(var));
		return "NULL".equals(type);
	}

	public boolean isVariableInRWorkspace(String var) {
		RBool test = evalAndCheck("\"" + var + "\"" + " %in% objects()")
				.asBool();
		if (test == null) {
			throw new RLinkErrorException("REXP convertion to RBool error");
		}
		return test.isTRUE();
	}

	public String getRandomVariable() {
		String result = null;
		boolean done = false;
		while (!done) {
			result = RCodeGenerator.getTemporaryVariable();
			done = !isVariableInRWorkspace(result);
		}
		return result;
	}

	public boolean removeRVariable(String var) {
		return isVariableInRWorkspace(var) ? eval("rm(" + var + ")") : true;
	}

	public String[] getWorkspaceVarNames() {
		return evalGetStringArray("objects()");
	}

	public String getRObjectType(String obj) {
		return evalAndCheck(RCodeGenerator.rGetType(obj)).asString();
	}

	public String getRObjectClass(String obj) {
		return evalAndCheck(RCodeGenerator.rGetClass(obj)).asString();
	}

	public boolean hasAttributes(String var) {
		return !isNULL("attributes(" + var + ")");
	}

	public synchronized boolean isValidRSession() {
		return engine.waitForR();
	}

	public synchronized void stopR() {
		engine.end();

	}

	public synchronized void startInteractiveConsole() {
		engine.startMainLoop();
	}

	public int[] getMissingElementPositions(String varNameOrCode) {
		return this.evalGetIntArray("which(is.na(" + varNameOrCode
				+ ") & !is.nan(" + varNameOrCode + "))");
	}

	public int[] getNaNElementPositions(String varNameOrCode) {
		return this.evalGetIntArray("which(is.nan(" + varNameOrCode + "))");
	}

	public int[] getPositiveInfinityPositions(String varNameOrCode) {
		return this.evalGetIntArray("which(" + varNameOrCode + " == Inf)");
	}

	public int[] getNegativeInfinityPositions(String varNameOrCode) {
		return this.evalGetIntArray("which(" + varNameOrCode + " == - Inf)");
	}

	public int[] getComplexInfinityPositions(String varNameOrCode) {
		return this.evalGetIntArray("which(is.infinite(" + varNameOrCode
				+ ") & !(" + varNameOrCode + " == Inf) & !(" + varNameOrCode
				+ " == -Inf ))");

	}

	public String deparse(String varNameOrCode) {
		return evalAndCheck("deparse(" + varNameOrCode + ")").asString();
	}

}
