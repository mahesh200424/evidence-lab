import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import core


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.data=patch.object(core,'DATA',Path(self.tmp.name)); self.data.start()
        self.db=patch.object(core,'DB',Path(self.tmp.name)/'test.sqlite3'); self.db.start()

    def tearDown(self):
        self.data.stop(); self.db.stop(); self.tmp.cleanup()

    def test_page_preservation_duplicate_and_ranking(self):
        with patch.object(core,'embed',return_value=[[1.,0.],[0.,1.]]):
            core.index_pages('paper.pdf',['Cats are mammals.','Birds have feathers.'])
            self.assertTrue(core.index_pages('paper.pdf',['Cats are mammals.','Birds have feathers.'])['duplicate'])
        with patch.object(core,'embed',return_value=[[0.,1.]]):
            evidence=core.retrieve('What has feathers?',1)
        self.assertEqual(evidence[0]['page'],2)
        self.assertEqual(evidence[0]['name'],'paper.pdf')
        self.assertAlmostEqual(evidence[0]['score'],1.)

    def test_no_readable_text_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'OCR'):
            core.index_pages('scan.pdf',['   '])

    def test_unknown_citation_is_withheld(self):
        sources=[{'citation':'S1','text':'Evidence','name':'paper','page':1}]
        response={'message':{'content':json.dumps({'answer':'Unsupported fact','citations':['S9'],'insufficient_evidence':False})}}
        with patch.object(core,'models',return_value=['test']),patch.object(core,'retrieve',return_value=sources),patch.object(core,'api',return_value=response):
            result=core.answer('Question','test')
        self.assertTrue(result['citation_check_rejected'])
        self.assertTrue(result['insufficient_evidence'])
        self.assertEqual(result['citations'],[])

    def test_metrics_exclude_unreviewed_and_failed(self):
        result=core.metrics([{'seconds':2,'retrieval_hit':True,'abstention_correct':False,'human_correct':None},
            {'seconds':4,'retrieval_hit':False,'abstention_correct':True,'human_correct':True}, {'error':'offline'}])
        self.assertEqual(result['retrieval_hit_percent'],50.)
        self.assertEqual(result['human_correct_percent'],100.)
        self.assertEqual(result['failed'],1)
        self.assertEqual(result['mean_seconds'],3.)

    def test_invalid_evaluation_rejected(self):
        with self.assertRaises(ValueError): core.validate_cases([{'question':'q','answerable':'false'}])

    def test_pdf_extraction_and_source_storage(self):
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
        writer=PdfWriter()
        page=writer.add_blank_page(width=612,height=792)
        font=DictionaryObject({NameObject('/Type'):NameObject('/Font'),NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/Helvetica')})
        page[NameObject('/Resources')]=DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):writer._add_object(font)})})
        stream=DecodedStreamObject(); stream.set_data(b'BT /F1 12 Tf 72 720 Td (Local PDF extraction test.) Tj ET')
        page[NameObject('/Contents')]=writer._add_object(stream)
        output=io.BytesIO(); writer.write(output)
        with patch.object(core,'embed',return_value=[[1.,0.]]):
            result=core.index_pdf('fixture.pdf',output.getvalue())
        self.assertEqual(result['chunks'],1)
        doc=core.documents()[0]
        self.assertTrue((core.DATA/doc['file']).is_file())
        with patch.object(core,'embed',return_value=[[1.,0.]]):
            evidence=core.retrieve('PDF',1)[0]
        self.assertIn('Local PDF extraction test.',evidence['text'])
        self.assertEqual(evidence['page'],1)


if __name__=='__main__': unittest.main()
