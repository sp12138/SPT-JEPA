import unittest
from as_jepa_tools.full_protocol import choose,phase_decision


class FullProtocolContracts(unittest.TestCase):
    def row(self,name,beta,score,rate,plan):
        return dict(id=name,beta=beta,score=score,rate=rate,planning=plan)

    def test_generation_then_margin_then_rate_in_the_spec_order(self):
        rows=[self.row("a",1.,.5,3.,80.),self.row("b",4.,.53,1.,65.),self.row("c",16.,.54,2.,75.),self.row("d",64.,.8,.1,99.)]
        self.assertEqual(choose(rows)["winner"]["id"],"c")
        rows[1]["planning"]=70.
        self.assertEqual(choose(rows)["winner"]["id"],"b")
        rows[1]["planning"]=None
        self.assertEqual(choose(rows)["state"],"waiting_for_tied_planning")

    def trial(self,name,beta,score,collapsed=False):
        return dict(id=name,beta=beta,stage3=dict(collapsed=collapsed),
                    final=dict(collapsed=False,rate_nats_per_frame=beta,active_dims=64,state_dim=64),
                    score8=dict(h7_agent_ratio=score*1.2,h7_block_ratio=score*1.2),
                    score10=dict(h7_agent_ratio=score,h7_block_ratio=score),planning=50.)

    def test_an_edge_price_requires_extension_before_selection(self):
        trials=[self.trial("a",1.,.2),self.trial("b",4.,.5),self.trial("c",16.,.6)]
        result=phase_decision(trials,True)
        self.assertEqual(result["state"],"edge_extension_required")
        self.assertEqual(result["requested_beta"],.25)
        trials.insert(0,self.trial("z",.25,.8))
        result=phase_decision(trials,True)
        self.assertEqual(result["state"],"selected")
        self.assertGreater(result["winner"]["continuation_fraction"],.05)

    def test_missing_results_and_three_epoch_failures_are_distinct(self):
        trials=[self.trial("a",1.,.4),self.trial("b",4.,.5)]
        trials[0]["stage3"]=None
        self.assertEqual(phase_decision(trials,False)["state"],"waiting_for_stage3")
        trials=[self.trial("a",1.,.4,True),self.trial("b",4.,.5,True)]
        self.assertEqual(phase_decision(trials,False)["state"],"all_collapsed")
        trials[0]["stage3"]["collapsed"]=False;trials[0]["score10"]=None
        self.assertEqual(phase_decision(trials,False)["state"],"waiting_for_full_budget")


if __name__=="__main__":
    unittest.main()
