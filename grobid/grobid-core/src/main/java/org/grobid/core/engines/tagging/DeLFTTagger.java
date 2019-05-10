package org.grobid.core.engines.tagging;

import com.google.common.base.Joiner;
import org.grobid.core.GrobidModel;
import org.grobid.core.GrobidModels;
import org.grobid.core.jni.DeLFTModel;

import java.io.IOException;

/**
 * 
 * @author: Patrice
 */
public class DeLFTTagger implements GenericTagger {

    private final DeLFTModel delftModel;

    public DeLFTTagger(GrobidModel model) {
        delftModel = new DeLFTModel(model);
    }

    @Override
    public String label(Iterable<String> data) {
        return label(Joiner.on('\n').join(data));
    }

    @Override
    public String label(String data) {
        return delftModel.label(data);
    }

    @Override
    public void close() throws IOException {
        delftModel.close();
    }
}
